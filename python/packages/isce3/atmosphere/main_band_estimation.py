import journal

import numpy as np

from .ionosphere_estimation import IonosphereEstimation
from isce3.signal.interpolate_by_range import decimate_freq_a_array
from isce3.unwrap.preprocess import interpret_subswath_mask


class MainBandIonosphereEstimation(IonosphereEstimation):
    '''Virtual class for main band ionosphere estimation methods
    '''
    def __init__(self,
                 main_center_freq=None,
                 side_center_freq=None,
                 low_center_freq=None,
                 high_center_freq=None,
                 slant_main=None,
                 slant_side=None):
        """Initialized IonosphereEstimation Class

        Parameters
        ----------
        main_center_freq : float
            center frequency of main band (freqA) [Hz]
        side_center_freq : float
            center frequency of side band (freqB) [Hz]
        low_center_freq : float
            center frequency of lower sub-band of the main band [Hz]
        high_center_freq : float
            center frequency of upper sub-band of the main band [Hz]
        """
        super().__init__(main_center_freq, side_center_freq, low_center_freq,
                         high_center_freq)

        self.estimate_iono = None
        self.estimate_sigma = None
        self.compute_unwrap_err = None

    def compute_disp_nondisp(
            self,
            phi_sub_low=None,
            phi_sub_high=None,
            phi_diff_low_high=None,
            phi_main=None,
            phi_side=None,
            phi_diff_ms=None,
            slant_main=None,
            slant_side=None,
            comm_unwcor_coef=None,
            diff_unwcor_coef=None,
            no_data=0):

        """Estimates dispersive and non-dispersive phase using given
        spectral diversity method. Note that each methods require different
        unwrapped interferograms.
        - split_main_band requires [phi_sub_low, phi_sub_high]
        - main_side_band requires [phi_main, phi_side]
        - main_diff_ms_band requires [phi_main, phi_side]
        If unwrapping correction terms are given, unwrapped phase array
        are corrected.

        Parameters
        ----------
        phi_sub_low : numpy.ndarray
            unwrapped phase array of low sub-band interferogram
        phi_sub_high : numpy.ndarray
            unwrapped phase array of high sub-band interferogram
        phi_diff_low_high : numpy.ndarray
            unwrapped phase array of difference between low and
            high sub-band interferograms
        phi_main : numpy.ndarray
            unwrapped phase array of frequency A interferogram
        phi_side : numpy.ndarray
            unwrapped phase array of frequency B interferogram
        slant_main : numpy.ndarray
            slant range array of frequency A interferogram
        slant_side : numpy.ndarray
            slant range array of frequency B interferogram
        comm_unwcor_coef : numpy.ndarray
            common correction coefficient of unwrapped phases
        diff_unwcor_coef : numpy.ndarray
            differential correction coefficient of unwrapped phases
        no_data : float
            no data value

        Returns
        -------
        dispersive : numpy.ndarray
            numpy array of dispersive array
        non_dispersive : numpy.ndarray
            non-dispersive phase array
        """
        error_channel = journal.error(
            'MainBandIonosphereEstimation.compute_disp_nondisp')

        # Validate mandatory inputs for main/side-band or main/diff-MS
        if phi_main is None or phi_side is None:
            err_str = "unwrapped interferogram array main and side band"\
                "is required."
            error_channel.log(err_str)
            raise ValueError(err_str)
        # Match spatial sampling by decimating main-band to side-band spacing
        phi_main = decimate_freq_a_array(slant_main,
                                         slant_side,
                                         phi_main)
        if phi_diff_ms is not None:
            no_data_array = (phi_main == no_data) |\
                            (phi_diff_ms == no_data)
        else:
            no_data_array = (phi_main == no_data) |\
                            (phi_side == no_data)

        # correct unwrapped phase, if coefficients are supplied
        if comm_unwcor_coef is not None and diff_unwcor_coef is not None:
            if phi_diff_ms is not None:
                phi_main = phi_main - 2 * np.pi * comm_unwcor_coef
                phi_diff_ms = phi_diff_ms - 2 * np.pi * diff_unwcor_coef
            else:
                phi_main = phi_main - 2 * np.pi * comm_unwcor_coef
                phi_side = phi_side - 2 * np.pi *\
                    (comm_unwcor_coef + diff_unwcor_coef)
        # Estimate dispersive / non-dispersive components after correcting
        # unwrapped phase
        dispersive, non_dispersive = self.estimate_iono(
            f0=self.f0,
            f1=self.f1,
            phi0=phi_main,
            phi1=phi_side,
            phi_diff_ms=phi_diff_ms)

        dispersive = np.where(no_data_array, no_data, dispersive)
        non_dispersive = np.where(no_data_array, no_data, non_dispersive)
        return dispersive, non_dispersive

    def get_coherence_mask_array(
            self,
            main_array=None,
            side_array=None,
            diff_ms_array=None,
            low_band_array=None,
            high_band_array=None,
            diff_low_high_band_array=None,
            slant_main=None,
            slant_side=None,
            threshold=0.5):
        """Get mask from coherences

        Parameters
        ----------
        main_array : numpy.ndarray
            coherence of main-band interferogram
        side_array : numpy.ndarray
            coherence of side-band interferogram
        diff_ms_array : numpy.ndarray
            coherence of difference between main and side band
            interferograms
        low_band_array : numpy.ndarray
            coherence of main-band interferogram
        high_band_array : numpy.ndarray
            coherence of side-band interferogram
        diff_low_high_band_array : numpy.ndarray
            coherence of difference (high-low) interferogram
        slant_main : numpy.ndarray
            slant range array of frequency A band
        slant_side : numpy.ndarray
            slant range array of frequency B band
        threshold : float
            thresholds for coherence

        Returns
        -------
        mask_array : numpy.ndarray
            2D mask array extracted from coherence or
            connected components
            1: valid pixels,
            0: invalid pixels.
        """
        return self.get_mask_array(
            main_array=main_array,
            side_array=side_array,
            diff_ms_array=diff_ms_array,
            low_band_array=low_band_array,
            high_band_array=high_band_array,
            diff_low_high_band_array=diff_low_high_band_array,
            slant_main=slant_main,
            slant_side=slant_side,
            threshold=threshold)

    def get_conn_component_mask_array(
            self,
            main_array=None,
            side_array=None,
            diff_ms_array=None,
            low_band_array=None,
            high_band_array=None,
            diff_low_high_band_array=None,
            slant_main=None,
            slant_side=None):
        """Get mask from connected components

        Parameters
        ----------
        main_array : numpy.ndarray
            coherence of main-band interferogram
        side_array : numpy.ndarray
            coherence of side-band interferogram
        diff_ms_array : numpy.ndarray
            coherence of difference between main and side band
            interferograms
        low_band_array : numpy.ndarray
            coherence of main-band interferogram
        high_band_array : numpy.ndarray
            coherence of side-band interferogram
        diff_low_high_band_array : numpy.ndarray
            coherence of difference (high-low) interferogram
        slant_main : numpy.ndarray
            slant range array of frequency A band
        slant_side : numpy.ndarray
            slant range array of frequency B band

        Returns
        -------
        mask_array : numpy.ndarray
            2D mask array extracted from coherence or
            connected components
            1: valid pixels,
            0: invalid pixels.
        """
        return self.get_mask_array(
            main_array=main_array,
            side_array=side_array,
            diff_ms_array=diff_ms_array,
            low_band_array=low_band_array,
            high_band_array=high_band_array,
            diff_low_high_band_array=diff_low_high_band_array,
            slant_main=slant_main,
            slant_side=slant_side,
            threshold=0)

    def get_mask_array(
            self,
            main_array=None,
            side_array=None,
            diff_ms_array=None,
            low_band_array=None,
            high_band_array=None,
            diff_low_high_band_array=None,
            slant_main=None,
            slant_side=None,
            threshold=0.5):
        """Get mask from coherence

        Parameters
        ----------
        main_array : numpy.ndarray
            coherence of main-band interferogram
        side_array : numpy.ndarray
            coherence of side-band interferogram
        diff_ms_array : numpy.ndarray
            coherence of difference (main-side) interferogram
        low_band_array : numpy.ndarray
            coherence of low subband interferogram
        high_band_array : numpy.ndarray
            coherence of high subband interferogram
        diff_low_high_band_array : numpy.ndarray
            coherence of difference (high-low) interferogram
        slant_main : numpy.ndarray
            slant range array of frequency A band
        slant_side : numpy.ndarray
            slant range array of frequency B band
        threshold : float
            thresholds for coherence

        Returns
        -------
        mask_array : numpy.ndarray
            2D mask array extracted from coherence or
            connected components
            1: valid pixels,
            0: invalid pixels.
        """
        # Validate threshold
        if threshold < 0 or threshold >= 1:
            raise ValueError(
                f"threshold must be in the range [0, 1), got {threshold}"
            )
        # decimate coherence or connected components
        # when side array is also used.
        if side_array is not None:
            main_array = decimate_freq_a_array(
                slant_main,
                slant_side,
                main_array)
        if diff_ms_array is not None:
            mask_array = (main_array > threshold) & \
                         (diff_ms_array > threshold)
        else:
            mask_array = (main_array > threshold) & \
                         (side_array > threshold)

        return mask_array

    def get_valid_area(
            self,
            main_array=None,
            side_array=None,
            diff_ms_array=None,
            low_band_array=None,
            high_band_array=None,
            diff_low_high_band_array=None,
            slant_main=None,
            slant_side=None,
            invalid_value=0):
        """Build a boolean mask of valid pixels based on an invalid_value
        threshold and NaNs.
        A pixel is considered valid (True) if:
        - Its value in `main_array` is neither equal to `invalid_value` nor NaN
        - And, if `side_array` is provided, its value in `side_array` also
            is neither equal to `invalid_value` nor NaN.

        Parameters
        ----------
        main_array : numpy.ndarray
            image of main-band interferogram
        side_array : numpy.ndarray
            image of side-band interferogram
        diff_ms_array : numpy.ndarray
            image of difference between main and side
            interferogram
        low_band_array : numpy.ndarray
            image of low subband interferogram
        high_band_array : numpy.ndarray
            image of high subband interferogram
        diff_low_high_band_array : numpy.ndarray
            image of difference between low and high subbands
        slant_main : numpy.ndarray
            slant range array of frequency A band
        slant_side : numpy.ndarray
            slant range array of frequency B band
        invalid_value : float
            invalid_value

        Returns
        -------
        mask_array : numpy.ndarray
            2D binary array extracted from images
            1: valid pixels,
            0: invalid pixels.
        """
        # decimate image when side array is also used.
        if side_array is not None:
            main_array = decimate_freq_a_array(
                slant_main,
                slant_side,
                main_array)
        if diff_ms_array is not None:
            mask_array = (main_array != invalid_value) & \
                         (diff_ms_array != invalid_value) & \
                         ~np.isnan(main_array) & \
                         ~np.isnan(diff_ms_array)
        else:
            mask_array = (main_array != invalid_value) & \
                         (side_array != invalid_value) & \
                         ~np.isnan(main_array) & \
                         ~np.isnan(side_array)

        return mask_array

    def get_subswath_mask_array(
            self,
            main_array=None,
            side_array=None,
            low_band_array=None,
            high_band_array=None,
            slant_main=None,
            slant_side=None):
        """Get mask from subswath mask
        Parameters
        ----------
        main_array : numpy.ndarray
            subswath mask of main-band interferogram
        side_array : numpy.ndarray
            subswath mask of side-band interferogram
        low_band_array : numpy.ndarray
            subswath mask of high subband interferogram
        high_band_array : numpy.ndarray
            subswath mask of low subband interferogram
        slant_main : numpy.ndarray
            slant range array of frequency A band
        slant_side : numpy.ndarray
            slant range array of frequency B band
        Returns
        -------
        mask_array : numpy.ndarray
            2D mask array extracted from coherence or
            connected components
            1: valid pixels,
            0: invalid pixels.
        """
        # decimate subswath mask
        # when side array is also used.
        if side_array is not None:
            main_array = decimate_freq_a_array(
                slant_main,
                slant_side,
                main_array)
        main_band_reference, main_band_secondary, _ = \
            interpret_subswath_mask(main_array)
        side_band_reference, side_band_secondary, _ = \
            interpret_subswath_mask(side_array)
        main_valid_area = main_band_reference & main_band_secondary
        side_valid_area = side_band_reference & side_band_secondary

        # Combine both conditions using logical AND
        final_mask = main_valid_area & side_valid_area

        return final_mask

    def estimate_iono_std(
            self,
            main_coh=None,
            side_coh=None,
            diff_ms_coh=None,
            low_band_coh=None,
            high_band_coh=None,
            diff_low_high_coh=None,
            slant_main=None,
            slant_side=None,
            number_looks=1,
            resample_flag=True):
        """Calculate the theoretical standard deviation of
        the ionospheric phase based on the coherencess

        Parameters
        ----------
        main_coh : numpy.ndarray
            coherence of main-band interferogram
        side_coh : numpy.ndarray
            coherence of side-band interferogram
        diff_ms_coh : numpy.ndarray
            coherence of difference (main-side) interferogram
        low_band_coh : numpy.ndarray
            coherence of low subband interferogram
        high_band_coh : numpy.ndarray
            coherence of high subband interferogram
        diff_low_high_band_array : numpy.ndarray
            coherence of difference (high-low) interferogram
        slant_main : numpy.ndarray
            slant range array of frequency A band
        slant_side : numpy.ndarray
            slant range array of frequency B band
        number_looks : int
            number of looks
        resample_flag : bool

        Returns
        -------
        sig_phi_iono : numpy.ndarray
            phase standard deviation of ionosphere phase
        sig_nondisp : numpy.ndarray
            phase standard deviation of non-dispersive
        """
        error_channel = journal.error(
            'MainBandIonosphereEstimation.estimate_iono_std')
        if side_coh is None and diff_ms_coh is None:
            err_str = "estimate_iono_std requires at least one of "\
                "`diff_ms_coh` or `side_coh` to be provided."
            error_channel.log(err_str)
            raise ValueError(err_str)
        if main_coh is None:
            err_str = "estimate_iono_std requires `main_coh` to be provided."
            error_channel.log(err_str)
            raise ValueError(err_str)
        side_used_coh = diff_ms_coh if diff_ms_coh is not None else side_coh

        # resample coherences array of frequency A to
        # frequency B grid
        if resample_flag:
            if slant_main is None:
                slant_main = self.slant_main
            if slant_side is None:
                slant_side = self.slant_side

            main_coh = decimate_freq_a_array(
                slant_main,
                slant_side,
                main_coh)

        sig_phi_main = np.divide(
            np.sqrt(1 - main_coh**2),
            main_coh * np.sqrt(2 * number_looks),
            out=np.zeros_like(main_coh),
            where=main_coh != 0)

        # estimate sigma from main- and side- band coherences
        sig_phi_side = np.divide(
            np.sqrt(1 - side_used_coh**2),
            side_used_coh * np.sqrt(2 * number_looks),
            out=np.zeros_like(side_used_coh),
            where=side_used_coh != 0)

        sig_phi_iono, sig_nondisp = \
            self.estimate_sigma(
                sig_phi_main,
                sig_phi_side)

        return sig_phi_iono, sig_nondisp


    def estimate_sigma_main_side(
            self,
            sig_phi0,
            sig_phi1):
        """Estimate sigma from coherence for main-side method

        Parameters
        ----------
        sig_phi0 : numpy.ndarray
            phase standard deviation of main-band interferogram
        sig_phi1 : numpy.ndarray
            phase standard deviation of side-band interferogram

        Returns
        -------
        sig_iono : numpy.ndarray
            phase standard deviation of ionosphere phase
        sig_nondisp : numpy.ndarray
            2D array of phase standard deviation of non-dispersive
        """
        a = (self.f1**2) / (self.f1**2 - self.f0**2)
        b = (self.f0 * self.f1) / (self.f1**2 - self.f0**2)
        c = (self.f0) / (self.f1**2 - self.f0**2)

        sig_iono = np.sqrt(a**2 * sig_phi0**2 + b**2 * sig_phi1**2)
        sig_non_disp = np.sqrt(c**2 * (self.f0**2 * sig_phi0**2
                               + self.f1**2 * sig_phi1**2))

        return sig_iono, sig_non_disp

    def compute_unwrapp_error(
            self,
            disp_array,
            nondisp_array,
            main_runw=None,
            side_runw=None,
            diff_ms_runw=None,
            slant_main=None,
            slant_side=None,
            low_sub_runw=None,
            high_sub_runw=None,
            diff_low_high_runw=None):
        """Compute unwrapping error coefficients

        Parameters
        ----------
        disp_array : numpy.ndarray
            2D dispersive array estimated from given methods
        nondisp_array : numpy.ndarray
            2D non-dispersive array estimated from given methods
        main_runw : numpy.ndarray
            2D runw array of main-band interferogram
        side_runw : numpy.ndarray
            2D runw array of of side-band interferogram
        low_sub_runw : numpy.ndarray
            2D runw array of low sub-band interferogram
        high_sub_runw : numpy.ndarray
            2D runw array of high sub-band interferogram
        diff_low_high_runw : numpy.ndarray
            2D runw array of difference of high and low sub-band interferogram
        Returns
        -------
        com_unw_coeff : numpy.ndarray
            2D common unwrapping error coefficient array
        diff_unw_coeff : numpy.ndarray
            2D differential unwrapping error coefficient array
        """
        com_unw_coeff, diff_unw_coeff = \
            super().compute_unwrapp_error(
                disp_array=disp_array,
                nondisp_array=nondisp_array,
                compute_unwrapp_error_func=self.compute_unwrap_err,
                main_runw=main_runw,
                side_runw=side_runw,
                diff_ms_runw=diff_ms_runw,
                slant_main=slant_main,
                slant_side=slant_side,
                low_sub_runw=low_sub_runw,
                high_sub_runw=high_sub_runw,
                diff_low_high_runw=diff_low_high_runw)

        return com_unw_coeff, diff_unw_coeff


class MainSideBandIonosphereEstimation(MainBandIonosphereEstimation):
    '''Main side band ionosphere estimation
    '''
    def __init__(self,
                 main_center_freq=None,
                 side_center_freq=None,
                 low_center_freq=None,
                 high_center_freq=None,
                 method=None):
        """Initialized IonosphereEstimation Class

        Parameters
        ----------
        main_center_freq : float
            center frequency of main band (freqA) [Hz]
        side_center_freq : float
            center frequency of side band (freqB) [Hz]
        low_center_freq : float
            center frequency of lower sub-band of the main band [Hz]
        high_center_freq : float
            center frequency of upper sub-band of the main band [Hz]
        method : {'split_main_band', 'main_side_band',
            'main_diff_ms_band'}
            ionosphere estimation method
        """
        super().__init__(main_center_freq, side_center_freq, low_center_freq,
                         high_center_freq, method)

        self.estimate_iono = estimate_iono_main_side
        self.estimate_sigma = self.estimate_sigma_main_side
        self.compute_unwrap_err = compute_unwrapp_error_main_side_band


class MainDiffMsBandIonosphereEstimation(MainBandIonosphereEstimation):
    '''Main diff MS band ionosphere estimation
    '''
    def __init__(self,
                 main_center_freq=None,
                 side_center_freq=None,
                 low_center_freq=None,
                 high_center_freq=None,
                 method=None):
        """Initialized IonosphereEstimation Class

        Parameters
        ----------
        main_center_freq : float
            center frequency of main band (freqA) [Hz]
        side_center_freq : float
            center frequency of side band (freqB) [Hz]
        low_center_freq : float
            center frequency of lower sub-band of the main band [Hz]
        high_center_freq : float
            center frequency of upper sub-band of the main band [Hz]
        method : {'split_main_band', 'main_side_band',
            'main_diff_ms_band'}
            ionosphere estimation method
        """
        super().__init__(main_center_freq, side_center_freq, low_center_freq,
                         high_center_freq, method)

        self.estimate_iono = estimate_iono_main_diff
        self.estimate_sigma = self.estimate_sigma_main_side
        self.compute_unwrap_err = compute_unwrapp_error_main_diff_ms_band


def compute_unwrapp_error_main_diff_ms_band(
        f0,
        f1,
        disp_array,
        nondisp_array,
        main_runw,
        diff_ms_runw,
        side_runw=None,
        freq_low=None,
        freq_high=None,
        low_sub_runw=None,
        high_sub_runw=None,
        diff_low_high_runw=None):

    """Compute unwrapping error coefficients for main_diff_ms_band
    method.

    Parameters
    ----------
    f0 : float
        radar center frequency of frequency A band
    f1 : float
        radar center frequency of frequency B band
    disp_array : numpy.ndarray
        2D dispersive array estimated from given methods
    nondisp_array : numpy.ndarray
        2D non-dispersive array estimated from given methods
    main_runw : numpy.ndarray
        2D runw array of main-band interferogram
    side_runw : numpy.ndarray
        2D runw array of of side-band interferogram

    Returns
    -------
    com_unw_coeff : numpy.ndarray
        2D common unwrapping error coefficient array
    diff_unw_coeff : numpy.ndarray
        2D differential unwrapping error coefficient array
    """
    f_diff = f1 - f0

    com_unw_coeff = np.round(
        (main_runw - (nondisp_array + disp_array)) / (2 * np.pi))
    term1 = f_diff / f0 * nondisp_array
    term2 = f_diff / f1 * disp_array
    diff_unw_coeff = np.round(
        (diff_ms_runw + term1 - term2) / (2 * np.pi))

    return com_unw_coeff, diff_unw_coeff


def compute_unwrapp_error_main_side_band(
        f0,
        f1,
        disp_array,
        nondisp_array,
        main_runw,
        side_runw,
        diff_ms_runw=None,
        freq_low=None,
        freq_high=None,
        low_sub_runw=None,
        high_sub_runw=None,
        diff_low_high_runw=None):

    """Compute unwrapping error coefficients.

    Parameters
    ----------
    f0 : float
        radar center frequency of frequency A band
    f1 : float
        radar center frequency of frequency B band
    disp_array : numpy.ndarray
        2D dispersive array estimated from given methods
    nondisp_array : numpy.ndarray
        2D non-dispersive array estimated from given methods
    main_runw : numpy.ndarray
        2D runw array of main-band interferogram
    side_runw : numpy.ndarray
        2D runw array of of side-band interferogram

    Returns
    -------
    com_unw_coeff : numpy.ndarray
        2D common unwrapping error coefficient array
    diff_unw_coeff : numpy.ndarray
        2D differential unwrapping error coefficient array
    """

    # Unwrapping errors exists in side interferogram,
    # but not in main interferogram.
    diff_unw_coeff = np.round(((1 - f1 / f0)
        * nondisp_array + (1 - f0 / f1) * disp_array
        + side_runw - main_runw) / (2 * np.pi))
    # Unwrapping errors detected from both main and side
    # interferograms.
    com_unw_coeff = np.round((main_runw + side_runw
        - (1 + f1 / f0) * nondisp_array
        - (1 + f0 / f1) * disp_array
        - 2 * np.pi * diff_unw_coeff) / (4 * np.pi))

    return com_unw_coeff, diff_unw_coeff


def estimate_iono_main_diff(f0,
                            f1,
                            phi0,
                            phi_diff_ms,
                            phi1=None):

    """Estimates ionospheric phase from main-band
    and the difference of main and side-band interferograms

    Parameters
    ----------
    f0 : float
        radar center frequency of main band
    f1 : float
        radar center frequency of side band
    phi0 : numpy.ndarray
        numpy array of main interferogram
    phi_diff_ms : numpy.ndarray
        numpy array of difference between main and side interferogram

    Returns
    -------
    dispersive : numpy.ndarray
        numpy array of estimated dispersive
    non_dispersive : numpy.ndarray
        numpy array of estimated non-dispersive
    """
    y_size, x_size = phi0.shape
    d = np.ones((2, y_size * x_size))
    d[0, :] = phi0.flatten()
    d[1, :] = phi_diff_ms.flatten()

    # Coefficient matrix for solving dispersive and non-dispersive
    # components
    coeff_mat = np.ones((2, 2))
    coeff_mat[1, 0] = (f1 - f0) / f1
    coeff_mat[1, 1] = (f0 - f1) / f0
    coeff_mat1 = np.linalg.pinv(coeff_mat)
    output = np.dot(coeff_mat1, d)

    non_dispersive = output[1].reshape(y_size, x_size)
    dispersive = output[0].reshape(y_size, x_size)

    return dispersive, non_dispersive


def estimate_iono_main_side(
        f0,
        f1,
        phi0,
        phi1,
        phi_diff_ms=None):

    """Estimates ionospheric phase from frequency A and B
    interferograms

    Parameters
    ----------
    f0 : float
        radar center frequency of frequency A band
    f1 : float
        radar center frequency of frequency B band
    phi0 : numpy.ndarray
        numpy array of frequency A interferogram
    phi1 : numpy.ndarray
        numpy array of frequency B interferogram

    Returns
    -------
    dispersive : numpy.ndarray
        numpy array of estimated dispersive
    non_dispersive : numpy.ndarray
        numpy array of estimated non-dispersive
    """

    y_size, x_size = phi0.shape
    d = np.ones((2, y_size * x_size))
    d[0, :] = phi0.flatten()
    d[1, :] = phi1.flatten()

    # Coefficient matrix for solving dispersive and non-dispersive
    # components
    coeff_mat = np.ones((2, 2))
    coeff_mat[1, 0] = f1 / f0
    coeff_mat[1, 1] = f0 / f1

    coeff_mat1 = np.linalg.pinv(coeff_mat)
    output = np.dot(coeff_mat1, d)

    non_dispersive = output[0].reshape(y_size, x_size)
    dispersive = output[1].reshape(y_size, x_size)

    return dispersive, non_dispersive
