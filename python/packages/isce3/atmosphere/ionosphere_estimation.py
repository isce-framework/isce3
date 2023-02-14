import journal
import numpy as np
from scipy.ndimage import median_filter


class IonosphereEstimation:
    '''
    Base class used to estimate ionospheric phase screen
    Not for standalone use!
    '''
    def __init__(self,
                 main_center_freq=None,
                 side_center_freq=None,
                 low_center_freq=None,
                 high_center_freq=None,
                 slant_main=None,
                 slant_side=None):

        """Initialized IonosphererEstimation Base Class

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

        error_channel = journal.error('ionosphere.IonosphereEstimation')

        # Center frequency for frequency A is needed for all methods.
        if main_center_freq is None:
            err_str = f"Center frequency for frequency A "\
                f" is needed for {method}"
            error_channel.log(err_str)
            raise ValueError(err_str)

        self.f0 = main_center_freq
        self.f1 = side_center_freq
        self.freq_low = low_center_freq
        self.freq_high = high_center_freq
        self.slant_main = slant_main
        self.slant_side = slant_side

    def get_mask_median_filter(self,
            disp,
            looks,
            threshold,
            median_filter_size):
        """Get mask using median filter

        Parameters
        ----------
        disp : numpy.ndarray
            2D dispersive array
        looks : int
            number of looks
        threshold : float
            coherence threshold to be used for std calculation

        Returns
        -------
        mask_array : numpy.ndarray
            2D mask array extracted from coherence or
            connected components
            1: valid pixels,
            0: invalid pixels.
        """

        std_iono, _ = self.estimate_iono_std(
            main_coh=threshold,
            side_coh=threshold,
            low_band_coh=threshold,
            high_band_coh=threshold,
            number_looks=looks,
            resample_flag=False)

        mask_array = np.abs(disp - median_filter(disp,
            median_filter_size)) < 3 * std_iono

        return mask_array

    def compute_unwrapp_error(
            self,
            disp_array,
            nondisp_array,
            compute_unwrapp_error_func=None,
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
        compute_unwrapp_error_func : function
            unwrapping function from derived class
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
        # decimate coherences array of frequency A to
        # frequency B grid
        if side_runw is not None:
            main_runw = decimate_freq_a_array(
                slant_main,
                slant_side,
                main_runw)

            if low_sub_runw is not None:
                low_sub_runw = decimate_freq_a_array(
                    slant_main,
                    slant_side,
                    low_sub_runw)

            if high_sub_runw is not None:
                high_sub_runw = decimate_freq_a_array(
                    slant_main,
                    slant_side,
                    high_sub_runw)

        com_unw_coeff, diff_unw_coeff = \
            compute_unwrapp_error_func(
            f0=self.f0,
            f1=self.f1,
            freq_low=self.freq_low,
            freq_high=self.freq_high,
            disp_array=disp_array,
            nondisp_array=nondisp_array,
            low_sub_runw=low_sub_runw,
            high_sub_runw=high_sub_runw,
            main_runw=main_runw,
            side_runw=side_runw,)

        return com_unw_coeff, diff_unw_coeff

def decimate_freq_a_array(
        slant_main,
        slant_side,
        target_runw):
    """decimate target_runw of main band to have same size with side band
    assuming slant_main and slant_side are evenly spaced

    Parameters
    ----------
    slant_main : numpy.ndarray
        slant range array of frequency A band
    slant_side : numpy.ndarray
        slant range array of frequency B band
    target_runw : numpy.ndarray
        RUNW array of frequency A band
        width of target_runw should be same with length of slant_main

    Returns
    -------
    decimated_array : numpy.ndarray
        decimated RUNW array
    """
    _, width = target_runw.shape

    first_index = np.argmin(np.abs(slant_main - slant_side[0]))
    spacing_main = slant_main[1] - slant_main[0]
    spacing_side = slant_side[1] - slant_side[0]

    resampling_scale_factor = int(np.round(spacing_side / spacing_main))

    x_cand = np.arange(1, width + 1)

    # find the maximum of the multiple of resampling_scale_factor
    decimate_width_end = np.max(x_cand[x_cand % resampling_scale_factor == 0])
    decimated_array = target_runw[
        :, first_index:decimate_width_end:resampling_scale_factor]

    return decimated_array
