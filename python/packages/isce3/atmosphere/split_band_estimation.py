import journal

import numpy as np

from .ionosphere_estimation import IonosphereEstimation
from isce3.signal.interpolate_by_range import decimate_freq_a_array
from isce3.unwrap.preprocess import interpret_subswath_mask


class SplitBandIonosphereEstimation(IonosphereEstimation):
    '''Split band ionosphere estimation
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

        error_channel = journal.error('ionosphere.SplitBandEstimation')

        # Check if required center frequencies for sub-bands are present
        if low_center_freq is None or high_center_freq is None:
            err_str = "Center frequency for frequency A is needed."
            error_channel.log(err_str)
            raise ValueError(err_str)

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
        phi_main : numpy.ndarray
            unwrapped phase array of frequency A interferogram
        phi_diff_low_high : numpy.ndarray
            unwrapped phase array of double difference between low and high
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
        error_channel = journal.error('SplitBandEstimation.compute_disp_nondisp')

        # set up mask for areas where no-data values are located
        # For the main band + difference between low and high interferogram
        if phi_main is not None and phi_diff_low_high is not None:
            no_data_array = (phi_main == no_data) |\
                (phi_diff_low_high == no_data)

            if comm_unwcor_coef is not None and diff_unwcor_coef is not None:
                phi_main = phi_main - 2 * np.pi * comm_unwcor_coef
                phi_diff_low_high = phi_diff_low_high - 2 * np.pi *\
                    diff_unwcor_coef

        # For the split_main_band method
        elif phi_sub_high is not None and phi_sub_low is not None:
            no_data_array = (phi_sub_high == no_data) |\
                (phi_sub_low == no_data)

            # compute absolute phase jump between two low and high subband
            # correct it to the high subband
            diff_sub = np.nanmean(phi_sub_low - phi_sub_high)
            num_jump = (np.abs(diff_sub) + np.pi) // (2.0 * np.pi)
            phi_sub_high = phi_sub_high + 2.0 * np.pi * num_jump

            # correct unwrapped phase when estimated unwrapping error are given
            if comm_unwcor_coef is not None and diff_unwcor_coef is not None:
                phi_sub_low = phi_sub_low - 2 * np.pi * comm_unwcor_coef
                phi_sub_high = phi_sub_high - 2 * np.pi *\
                    (comm_unwcor_coef + diff_unwcor_coef)

        dispersive, non_dispersive = self.estimate_iono(
            f0=self.f0,
            freq_low=self.freq_low,
            freq_high=self.freq_high,
            phi0_low=phi_sub_low,
            phi0_high=phi_sub_high,
            phi0_main=phi_main,
            phi0_diff_low_high=phi_diff_low_high
            )

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
        """Build a boolean mask of valid pixels based on an invalid_value
        threshold and NaNs from coherence. This will be used to mask out
        the low coherence areas.

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
        # decimate coherence or connected components
        # when side array is also used.
        if side_array is not None:
            if slant_main is None:
                slant_main = self.slant_main
            if slant_side is None:
                slant_side = self.slant_side

            if low_band_array is not None:
                low_band_array = decimate_freq_a_array(
                    slant_main,
                    slant_side,
                    low_band_array)
            if high_band_array is not None:
                high_band_array = decimate_freq_a_array(
                    slant_main,
                    slant_side,
                    high_band_array)

        if main_array is not None and diff_low_high_band_array is not None:
            mask_array = (main_array > threshold) & \
                (diff_low_high_band_array > threshold)
        else:
            mask_array = (high_band_array > threshold) & \
                        (low_band_array > threshold)
        mask_array = self.remove_single_pixels(mask_array)
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
            if slant_main is None:
                slant_main = self.slant_main
            if slant_side is None:
                slant_side = self.slant_side

            if low_band_array is not None:
                low_band_array = decimate_freq_a_array(
                    slant_main,
                    slant_side,
                    low_band_array)
            if high_band_array is not None:
                high_band_array = decimate_freq_a_array(
                    slant_main,
                    slant_side,
                    high_band_array)

        high_band_reference, high_band_secondary, _ = \
            interpret_subswath_mask(high_band_array)
        low_band_reference, low_band_secondary, _ = \
            interpret_subswath_mask(low_band_array)

        high_valid_area = high_band_reference & high_band_secondary
        low_valid_area = low_band_reference & low_band_secondary

        # Combine both conditions using logical AND
        final_mask = high_valid_area & low_valid_area

        return final_mask

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
            image of main-band interferogram
        high_band_array : numpy.ndarray
            image of side-band interferogram
        diff_low_high_band_array : numpy.ndarray
            image of difference between low- and high-sub bands
        slant_main : numpy.ndarray
            slant range array of frequency A band
        slant_side : numpy.ndarray
            slant range array of frequency B band
        invalid_value : float
            invalid_value

        Returns
        -------
        mask_array : numpy.ndarray
            2D mask array extracted from data
            1: valid pixels,
            0: invalid pixels.
        """
        # decimate coherence or connected components
        # when side array is also used.
        if side_array is not None:
            if slant_main is None:
                slant_main = self.slant_main
            if slant_side is None:
                slant_side = self.slant_side

            if low_band_array is not None:
                low_band_array = decimate_freq_a_array(
                    slant_main,
                    slant_side,
                    low_band_array)
            if high_band_array is not None:
                high_band_array = decimate_freq_a_array(
                    slant_main,
                    slant_side,
                    high_band_array)
        if main_array is not None and diff_low_high_band_array is not None:
            mask_array = (diff_low_high_band_array != invalid_value) & \
                         (main_array != invalid_value) & \
                         np.invert(np.isnan(diff_low_high_band_array)) & \
                         np.invert(np.isnan(main_array))
        else:
            mask_array = (high_band_array != invalid_value) & \
                         (low_band_array != invalid_value) & \
                         ~np.isnan(high_band_array) & \
                         ~np.isnan(low_band_array)
        mask_array = self.remove_single_pixels(mask_array)

        return mask_array

    def estimate_iono_std(
            self,
            main_coh=None,
            side_coh=None,
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
        # resample coherences array of frequency A to
        # frequency B grid
        if (side_coh is not None) and (resample_flag):
            if slant_main is None:
                slant_main = self.slant_main
            if slant_side is None:
                slant_side = self.slant_side

            main_coh = decimate_freq_a_array(
                slant_main,
                slant_side,
                main_coh)

        sig_phi_diff = None
        sig_phi_main = None
        sig_phi_low = None
        sig_phi_high = None

        # estimate sigma from sub-band coherences
        if (main_coh is not None) and (diff_low_high_coh is not None):
            sig_phi_main = np.sqrt(1 - main_coh**2) / \
                main_coh / np.sqrt(2 * number_looks)
            sig_phi_diff = np.sqrt(1 - diff_low_high_coh**2) / \
                diff_low_high_coh / np.sqrt(2 * number_looks)

        elif (low_band_coh is not None) and (high_band_coh is not None):
            sig_phi_low = np.sqrt(1 - low_band_coh**2) / \
                low_band_coh / np.sqrt(2 * number_looks)
            sig_phi_high = np.sqrt(1 - high_band_coh**2) / \
                high_band_coh / np.sqrt(2 * number_looks)

        sig_phi_iono, sig_nondisp = \
            self.estimate_sigma(
                freq_center=self.f0,
                freq_low=self.freq_low,
                freq_high=self.freq_high,
                sig_phi_low=sig_phi_low,
                sig_phi_high=sig_phi_high,
                sig_phi0=sig_phi_main,
                sig_phi_diff=sig_phi_diff
            )

        return sig_phi_iono, sig_nondisp

    def estimate_sigma_split_main_band(
            self,
            sig_phi_low,
            sig_phi_high):
        """Estimate sigma from coherence for split_main_band method

        Parameters
        ----------
        sig_phi_low : numpy.ndarray
            phase standard deviation of low sub-band interferogram
        sig_phi_high : numpy.ndarray
            phase standard deviation of high sub-band interferogram

        Returns
        -------
        sig_iono : numpy.ndarray
            2D phase standard deviation of ionosphere phase
        sig_nondisp : numpy.ndarray
            2D array of phase standard deviation of non-dispersive
        """
        coeff = self.freq_low * self.freq_high / self.f0 /\
            (self.freq_high**2 - self.freq_low**2)
        sig_iono = np.sqrt(
            coeff**2 * (self.freq_high**2 * sig_phi_low**2
                        + self.freq_low**2 * sig_phi_high**2)
            )

        coef_non = self.f0 / (self.freq_high**2 - self.freq_low**2)

        sig_nondisp = np.sqrt(
            (coef_non**2) * (self.freq_low**2) *
            (sig_phi_low**2) + (coef_non**2) *
            (self.freq_high**2) * (sig_phi_high**2))

        return sig_iono, sig_nondisp

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
            2D runw array of difference bewteen low and high sub-band interferograms
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
                low_sub_runw=low_sub_runw,
                high_sub_runw=high_sub_runw,
                diff_low_high_runw=diff_low_high_runw)

        return com_unw_coeff, diff_unw_coeff


class LowHighSubbandIonosphereEstimation(SplitBandIonosphereEstimation):
    '''Ionosphere estimation from Low and High subbands
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

        self.estimate_iono = estimate_iono_low_high
        self.estimate_sigma = estimate_sigma_split_main_band
        self.compute_unwrap_err = compute_unwrapp_error_split_main_band


class MainDiffLowHighSubbandIonosphereEstimation(SplitBandIonosphereEstimation):
    '''Ionosphere estimation from main band and difference between
       low and high subbands
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

        self.estimate_iono = estimate_iono_main_diff_low_high
        self.estimate_sigma = estimate_sigma_main_diff_low_high
        self.compute_unwrap_err = compute_unwrapp_error_main_diff_low_high


def estimate_sigma_split_main_band(
        freq_low,
        freq_high,
        freq_center,
        sig_phi_low,
        sig_phi_high,
        sig_phi0=None,
        sig_phi_diff=None):
    """Estimate sigma from coherence for split_main_band method

    Parameters
    ----------
    sig_phi_low : numpy.ndarray
        phase standard deviation of low sub-band interferogram
    sig_phi_high : numpy.ndarray
        phase standard deviation of high sub-band interferogram

    Returns
    -------
    sig_iono : numpy.ndarray
        2D phase standard deviation of ionosphere phase
    sig_nondisp : numpy.ndarray
        2D array of phase standard deviation of non-dispersive
    """
    coeff = (freq_low * freq_high / freq_center /
             (freq_high**2 - freq_low**2))
    sig_iono = np.sqrt(
        coeff**2 * (freq_high**2 * sig_phi_low**2
                    + freq_low**2 * sig_phi_high**2)
                    )

    coef_non = freq_center / (freq_high**2 - freq_low**2)

    sig_nondisp = np.sqrt(
        (coef_non**2) * (freq_low**2) * (sig_phi_low**2)
        + (coef_non**2) * (freq_high**2) * (sig_phi_high**2)
        )

    return sig_iono, sig_nondisp


def estimate_sigma_main_diff_low_high(
        freq_low,
        freq_high,
        freq_center,
        sig_phi0,
        sig_phi_diff,
        sig_phi_low=None,
        sig_phi_high=None):
    """Estimate sigma from coherence for split_main_band method

    Parameters
    ----------
    sig_phi_low : numpy.ndarray
        phase standard deviation of low sub-band interferogram
    sig_phi_high : numpy.ndarray
        phase standard deviation of high sub-band interferogram

    Returns
    -------
    sig_iono : numpy.ndarray
        2D phase standard deviation of ionosphere phase
    sig_nondisp : numpy.ndarray
        2D array of phase standard deviation of non-dispersive
    """

    coeff_a = (freq_low * freq_high / freq_center /
               (freq_high + freq_low))
    coeff_b = (freq_low * freq_high / freq_center / 2 /
               (freq_high - freq_low))

    sig_iono = np.sqrt(coeff_a**2 * sig_phi0**2 +
                       coeff_b**2 * sig_phi_diff**2)

    sig_nondisp = np.sqrt(sig_phi0**2 / 4 + coeff_b**2 * sig_phi_diff**2)

    return sig_iono, sig_nondisp


def compute_unwrapp_error_main_diff_low_high(
        f0,
        freq_low,
        freq_high,
        disp_array,
        nondisp_array,
        diff_low_high_runw,
        main_runw,
        low_sub_runw=None,
        high_sub_runw=None,
        f1=None,
        side_runw=None,
        diff_ms_runw=None):
    """Compute unwrapping error coefficients.

    Parameters
    ----------
    f0 : float
        radar center frequency of frequency A band
    freq_low : float
        radar center frequency of lower sub-band
    freq_high : float
        radar center frequency of upper sub-band
    disp_array : numpy.ndarray
        2D dispersive array estimated from given methods
    nondisp_array : numpy.ndarray
        2D non-dispersive array estimated from given methods
    low_sub_runw : numpy.ndarray
        2D runw array of low sub-band interferogram
    high_sub_runw : numpy.ndarray
        2D runw array of high sub-band interferogram

    Returns
    -------
    com_unw_coeff : numpy.ndarray
        2D common unwrapping error coefficient array
    diff_unw_coeff : numpy.ndarray
        2D differential unwrapping error coefficient array
    """

    freq_diff = freq_high - freq_low
    freq_sum = freq_high + freq_low
    freq_multi = freq_high * freq_low

    x_coeff = freq_multi / f0 / freq_sum
    z_coeff = - freq_multi / 2 / f0 / freq_diff

    diff_unw_coeff = np.round((2 * nondisp_array / z_coeff -
                               disp_array / z_coeff * x_coeff -
                               diff_low_high_runw) / 2 / np.pi)
    com_unw_coeff = np.round((
        (nondisp_array + disp_array) / (2 * x_coeff + 1) -
        main_runw / 2) / np.pi)

    return com_unw_coeff, diff_unw_coeff


def compute_unwrapp_error_split_main_band(
        f0,
        freq_low,
        freq_high,
        disp_array,
        nondisp_array,
        low_sub_runw,
        high_sub_runw,
        f1=None,
        main_runw=None,
        side_runw=None,
        diff_ms_runw=None,
        diff_low_high_runw=None):
    """Compute unwrapping error coefficients.

    Parameters
    ----------
    f0 : float
        radar center frequency of frequency A band
    freq_low : float
        radar center frequency of lower sub-band
    freq_high : float
        radar center frequency of upper sub-band
    disp_array : numpy.ndarray
        2D dispersive array estimated from given methods
    nondisp_array : numpy.ndarray
        2D non-dispersive array estimated from given methods
    low_sub_runw : numpy.ndarray
        2D runw array of low sub-band interferogram
    high_sub_runw : numpy.ndarray
        2D runw array of high sub-band interferogram

    Returns
    -------
    com_unw_coeff : numpy.ndarray
        2D common unwrapping error coefficient array
    diff_unw_coeff : numpy.ndarray
        2D differential unwrapping error coefficient array
    """
    freq_diff = freq_high - freq_low
    freq_multi = freq_high * freq_low

    # Unwrapping errors exists in high-subband interferogram,
    # but not in low-subband interferogram.
    diff_unw_coeff = np.round((
        (high_sub_runw) - (low_sub_runw)
        - (freq_diff / f0) * nondisp_array
        + (f0 * freq_diff / freq_multi) * disp_array) /
            2.0 / np.pi)
    # Unwrapping errors exists in both high and low-subband
    # interferograms.
    com_unw_coeff = np.round(
        (low_sub_runw + high_sub_runw
         - 2.0 * nondisp_array - 2.0 * disp_array) / 4.0 / np.pi
        - diff_unw_coeff / 2)

    return com_unw_coeff, diff_unw_coeff


def estimate_iono_low_high(
        f0,
        freq_low,
        freq_high,
        phi0_low,
        phi0_high,
        phi0_main=None,
        phi0_diff_low_high=None):

    """Estimates ionospheric phase from low and high sub-band
    interferograms i.e. split_main_band method

    Parameters
    ----------
    f0 : float
        radar center frequency of frequency A band
    freq_low : float
        radar center frequency of lower sub-band
    freq_high : float
        radar center frequency of upper sub-band
    phi0_low : numpy.ndarray
        numpy array of lower sub-band interferogram
    phi0_high : numpy.ndarray
        numpy array of upper sub-band interferogram

    Returns
    -------
    dispersive : numpy.ndarray
        numpy array of estimated dispersive
    non_dispersive : numpy.ndarray
        numpy array of estimated non-dispersive
    """
    if phi0_low.shape != phi0_high.shape:
        raise ValueError("phi0_low and phi0_high must have identical shapes")
    # Coefficient matrix for solving dispersive and non-dispersive
    # components
    a = freq_low / f0
    b = f0 / freq_low
    c = freq_high / f0
    d = f0 / freq_high

    det = a * d - b * c

    if det == 0:
        raise ZeroDivisionError("Frequency combination leads to singular matrix")

    # rows of A⁻¹
    m11 = d / det
    m12 = -b / det
    m21 = -c / det
    m22 = a / det

    # Compute outputs

    non_dispersive = m11 * phi0_low + m12 * phi0_high
    dispersive = m21 * phi0_low + m22 * phi0_high

    return dispersive, non_dispersive


def estimate_iono_main_diff_low_high(
        f0,
        freq_low,
        freq_high,
        phi0_main,
        phi0_diff_low_high,
        phi0_low=None,
        phi0_high=None):

    """Estimates ionospheric phase from low and high sub-band
    interferograms by the split_main_band method

    Parameters
    ----------
    f0 : float
        radar center frequency of frequency A band
    freq_low : float
        radar center frequency of low sub-band
    freq_high : float
        radar center frequency of high sub-band
    phi0_low : numpy.ndarray
        numpy array of low sub-band interferogram
    phi0_high : numpy.ndarray
        numpy array of high sub-band interferogram
    phi0_diff_low_high
    : numpy.ndarray
        numpy array of difference between low and high sub-band interferogram
    Returns
    -------
    dispersive : numpy.ndarray
        numpy array of estimated dispersive
    non_dispersive : numpy.ndarray
        numpy array of estimated non-dispersive
    """
    y_size, x_size = phi0_diff_low_high.shape
    x_coeff = freq_low * freq_high / f0 / (freq_low + freq_high)
    z_coeff = - freq_low * freq_high / f0 / (freq_high - freq_low) / 2

    dispersive = x_coeff * phi0_main + z_coeff * phi0_diff_low_high
    non_dispersive = phi0_main / 2 + z_coeff * phi0_diff_low_high

    return dispersive, non_dispersive
