import journal

import numpy as np

from .ionosphere_estimation import IonosphereEstimation
from isce3.signal.interpolate_by_range import decimate_freq_a_array

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
        """Initialized IonosphererEstimation Class

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

    def compute_disp_nondisp(self,
            phi_sub_low=None,
            phi_sub_high=None,
            phi_main=None,
            phi_side=None,
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

        if phi_sub_high is None:
            err_str = "upper sub-band unwrapped interferogram "\
                "is required for split_main_band method."
            error_channel.log(err_str)
            raise ValueError(err_str)

        if phi_sub_low is None:
            err_str = "lower sub-band unwrapped interferogram "\
                "is required for split_main_band method."
            error_channel.log(err_str)
            raise ValueError(err_str)

        # set up mask for areas where no-data values are located
        no_data_array = (phi_sub_high==no_data) |\
            (phi_sub_low==no_data)

        # correct unwrapped phase when estimated unwrapping error are given
        if comm_unwcor_coef is not None and diff_unwcor_coef is not None:
            phi_sub_low = phi_sub_low - 2 * np.pi * comm_unwcor_coef
            phi_sub_high = phi_sub_high - 2 * np.pi *\
                (comm_unwcor_coef + diff_unwcor_coef)

        dispersive, non_dispersive = estimate_iono_low_high(
            f0=self.f0,
            freq_low=self.freq_low,
            freq_high=self.freq_high,
            phi0_low=phi_sub_low,
            phi0_high=phi_sub_high)

        dispersive[no_data_array] = no_data
        non_dispersive[no_data_array] = no_data

        return dispersive, non_dispersive

    def get_coherence_mask_array(self,
            main_array=None,
            side_array=None,
            low_band_array=None,
            high_band_array=None,
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
        low_band_array : numpy.ndarray
            coherencen of main-band interferogram
        high_band_array : numpy.ndarray
            coherence of side-band interferogram
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
        return self.get_mask_array(main_array, side_array, low_band_array,
                                   high_band_array, slant_main, slant_side,
                                   threshold)

    def get_conn_component_mask_array(self,
            main_array=None,
            side_array=None,
            low_band_array=None,
            high_band_array=None,
            slant_main=None,
            slant_side=None):
        """Get mask from connected components

        Parameters
        ----------
        main_array : numpy.ndarray
            coherence of main-band interferogram
        side_array : numpy.ndarray
            coherence of side-band interferogram
        low_band_array : numpy.ndarray
            coherencen of main-band interferogram
        high_band_array : numpy.ndarray
            coherence of side-band interferogram
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
        return self.get_mask_array(main_array, side_array, low_band_array,
                                   high_band_array, slant_main, slant_side,
                                   0)

    def get_mask_array(self,
            main_array=None,
            side_array=None,
            low_band_array=None,
            high_band_array=None,
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
        low_band_array : numpy.ndarray
            coherencen of main-band interferogram
        high_band_array : numpy.ndarray
            coherence of side-band interferogram
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

        mask_array = (high_band_array > threshold) & \
                     (low_band_array > threshold)

        return mask_array

    def estimate_iono_std(
            self,
            main_coh=None,
            side_coh=None,
            low_band_coh=None,
            high_band_coh=None,
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
            coherencen of main-band interferogram
        high_band_coh : numpy.ndarray
            coherence of side-band interferogram
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

        # estimate sigma from sub-band coherences
        if (low_band_coh is not None) and (high_band_coh is not None):
            sig_phi_low = np.sqrt(1 - low_band_coh**2) / \
                low_band_coh / np.sqrt(2 * number_looks)
            sig_phi_high = np.sqrt(1 - high_band_coh**2) / \
                high_band_coh / np.sqrt(2 * number_looks)

        sig_phi_iono, sig_nondisp = \
            self.estimate_sigma_split_main_band(
            sig_phi_low,
            sig_phi_high)

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
        sig_iono = np.sqrt(coeff**2 * (self.freq_high**2 * sig_phi_low**2
            + self.freq_low**2 * sig_phi_high**2))

        coef_non = self.f0 / (self.freq_high**2 - self.freq_low**2)

        sig_nondisp = np.sqrt((coef_non**2) * (self.freq_low**2) *\
                (sig_phi_low**2) + (coef_non**2) *\
                (self.freq_high**2) * (sig_phi_high**2))

        return sig_iono, sig_nondisp

    def compute_unwrapp_error(
            self,
            disp_array,
            nondisp_array,
            main_runw=None,
            side_runw=None,
            slant_main=None,
            slant_side=None,
            low_sub_runw=None,
            high_sub_runw=None):
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
            compute_unwrapp_error_func=compute_unwrapp_error_split_main_band,
            main_runw=main_runw,
            side_runw=side_runw,
            low_sub_runw=low_sub_runw,
            high_sub_runw=high_sub_runw)

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
        side_runw=None):

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

    diff_unw_coeff = np.round(((high_sub_runw) - (low_sub_runw)\
        - (freq_diff / f0) * nondisp_array \
        + ( f0 * freq_diff / freq_multi) * disp_array) /\
            2.0 / np.pi)
    com_unw_coeff = np.round((low_sub_runw + high_sub_runw \
        - 2.0 * nondisp_array - 2.0 * disp_array ) / 4.0 / np.pi\
        - diff_unw_coeff / 2)

    return com_unw_coeff, diff_unw_coeff

def estimate_iono_low_high(
        f0,
        freq_low,
        freq_high,
        phi0_low,
        phi0_high):

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

    y_size, x_size = phi0_low.shape
    d = np.ones((2, y_size * x_size))
    d[0,:] = phi0_low.flatten()
    d[1,:] = phi0_high.flatten()
    coeff_mat = np.ones((2, 2))

    #import ipdb; ipdb.set_trace()
    coeff_mat[0, 0] = freq_low / f0
    coeff_mat[0, 1] = f0 / freq_low
    coeff_mat[1, 0] = freq_high / f0
    coeff_mat[1, 1] = f0 / freq_high
    coeff_mat1 = np.linalg.pinv(coeff_mat)
    output = np.dot(coeff_mat1, d)

    non_dispersive = output[0, :].reshape(y_size, x_size)
    dispersive = output[1].reshape(y_size, x_size)

    return dispersive, non_dispersive
