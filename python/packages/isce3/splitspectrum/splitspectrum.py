import math

from dataclasses import dataclass
import journal
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import resample

import isce3
from nisar.workflows.focus import cosine_window


@dataclass(frozen=True)
class bandpass_meta_data:
    # slant range spacing
    rg_pxl_spacing: float
    # wavelength
    wavelength: float
    # sampling frequency
    rg_sample_freq: float
    # bandwidth
    rg_bandwidth: float
    # center frequency
    center_freq: float
    # slant range
    slant_range: 'method'

    @classmethod
    def load_from_slc(cls, slc_product, freq):
        """Get meta data from SLC object.
        Parameters
        ----------
        slc_product : nisar.products.readers.SLC
            slc object
        freq : {'A', 'B'}
            frequency band
        Returns
        -------
        meta_data : bandpass_meta_data
            bandpass meta data object
        """
        rdr_grid = slc_product.getRadarGrid(freq)
        rg_sample_freq = \
            isce3.core.speed_of_light * 0.5 /\
            rdr_grid.range_pixel_spacing
        is_close = math.isclose(rg_sample_freq,
                                np.round(rg_sample_freq),
                                rel_tol=1e-8)
        if is_close:
            rg_sample_freq = np.round(rg_sample_freq)

        rg_bandwidth = \
            slc_product.getSwathMetadata(freq).processed_range_bandwidth
        center_frequency = \
            isce3.core.speed_of_light / rdr_grid.wavelength
        return cls(rdr_grid.range_pixel_spacing, rdr_grid.wavelength,
                   rg_sample_freq, rg_bandwidth, center_frequency,
                   rdr_grid.slant_range)


def check_range_bandwidth_overlap(ref_slc, sec_slc, pols):
    """Check if bandpass is needed.

    If the two SLCs differ in center frequency or bandwidth, then
    one SLC shall be bandpassed to a common frequency band. If
    necessary, determine which SLC will be bandpassed

    Parameters
    ----------
    ref_slc : nisar.products.readers.SLC
        Reference SLC object
    sec_slc : nisar.products.readers.SLC
        Secondary SLC object
    pols : dict
        Dict keying frequency ('A' or 'B') to list of polarization values.

    Returns
    -------
    mode : dict
        Dict mapping frequency band (e.g. "A" or "B") to
        SLC to be bandpassed ("ref" or "sec").
    """
    mode = dict()

    for freq, pol_list in pols.items():
        ref_meta_data = bandpass_meta_data.load_from_slc(ref_slc, freq)
        sec_meta_data = bandpass_meta_data.load_from_slc(sec_slc, freq)

        ref_wvl = ref_meta_data.wavelength
        sec_wvl = sec_meta_data.wavelength
        ref_bw = ref_meta_data.rg_bandwidth
        sec_bw = sec_meta_data.rg_bandwidth

        # check if two SLCs have same bandwidth and center frequency
        if (ref_wvl != sec_wvl) or (ref_bw != sec_bw):
            if ref_bw > sec_bw:
                mode[freq] = 'ref'
            else:
                mode[freq] = 'sec'
    return mode


class SplitSpectrum:
    '''
    Split the slant range spectrum
    '''

    def __init__(self,
                 rg_sample_freq,
                 rg_bandwidth,
                 center_frequency,
                 slant_range,
                 freq,
                 sampling_bandwidth_ratio=None):
        """Initialized Bandpass Class with SLC meta data

        Parameters
        ----------
        rg_sample_freq : float
            range sampling frequency
        rg_bandwidth : float
            range bandwidth [Hz]
        center_frequency : float
            center frequency of SLC [Hz]
        slant_range : new center frequency for bandpass [Hz]
        freq : {'A', 'B'}
            frequency band
        sampling_bandwidth_ratio: float
            The ratio of range sampling frequency to bandwidth.
            If not provided, sampling frequency will be same as
            input.
        """
        self.freq = freq
        self.rg_sample_freq = rg_sample_freq
        self.rg_pxl_spacing = \
            isce3.core.speed_of_light * 0.5 / self.rg_sample_freq
        self.rg_bandwidth = rg_bandwidth
        self.center_frequency = center_frequency
        self.slant_range = slant_range
        if sampling_bandwidth_ratio is None:
            sampling_bandwidth_ratio = rg_sample_freq / rg_bandwidth
        self.sampling_bandwidth_ratio = sampling_bandwidth_ratio

    def bandpass_shift_spectrum(self,
                                slc_raster,
                                low_frequency,
                                high_frequency,
                                new_center_frequency,
                                window_function,
                                window_shape=0.25,
                                fft_size=None,
                                resampling=True
                                ):

        """Bandpass SLC for a given bandwidth and shift the bandpassed
        spectrum to a new center frequency

        Parameters
        ----------
        slc_raster : numpy.ndarray
            numpy array of slc raster,
        low_frequency : float
            low  frequency of band to be passed [Hz]
        high_frequency : float
            high frequency band to be passed [Hz]
        new_center_frequency : float
            new center frequency for new bandpassed slc [Hz]
        window_function : str
            window type {tukey, kaiser, cosine}
        window_shape : float
            parameter for the raised cosine filter (e.g. 0 ~ 1)
        fft_size : int
            fft size.
        resampling : bool
            if True, then resample SLC and meta data with new range spacing
            If False, return SLC and meta with original range spacing

        Returns
        -------
        resampled_slc or slc_demodulate: numpy.ndarray
            numpy array of bandpassed slc
            if resampling is True,
            return resampled slc with bandpass and demodulation
            if resampling is False,
            return slc with bandpass and demodulation without resampling
        meta : dict
            dict containing meta data of bandpassed slc
            center_frequency, rg_bandwidth, range_spacing, slant_range
        """

        rg_sample_freq = self.rg_sample_freq
        rg_bandwidth = self.rg_bandwidth
        diff_frequency = self.center_frequency - new_center_frequency
        height, width = slc_raster.shape
        slc_raster = np.asanyarray(slc_raster, dtype='complex')

        slc_bp = self.bandpass_spectrum(
                          slc_raster=slc_raster,
                          low_frequency=low_frequency,
                          high_frequency=high_frequency,
                          window_function=window_function,
                          window_shape=window_shape,
                          fft_size=fft_size,
                          )

        # demodulate the SLC to be baseband to new center frequency
        # if fft_size > width, then crop the spectrum from 0 to width
        slc_demodulate = self.demodulate_slc(slc_bp[:, :width],
                                             diff_frequency,
                                             rg_sample_freq)

        # update metadata with new parameters
        meta = dict()
        new_bandwidth = high_frequency - low_frequency
        new_rg_sample_freq = np.abs(new_bandwidth) * \
            self.sampling_bandwidth_ratio

        meta['center_frequency'] = new_center_frequency
        meta['rg_bandwidth'] = new_bandwidth
        meta['rg_sample_freq'] = new_rg_sample_freq

        # Resampling changes the spacing and slant range
        if resampling:
            # due to the precision of the floating point, the resampling
            # scaling factor may be not integer.
            resampling_scale_factor = rg_sample_freq / new_rg_sample_freq

            # convert to integer
            if rg_sample_freq % new_rg_sample_freq == 0:
                resampling_scale_factor = np.round(resampling_scale_factor)
            else:
                err_msg = 'Resampling scaling factor ' \
                          f'{resampling_scale_factor} must be an integer.'
                raise ValueError(err_msg)

            sub_width = int(width / resampling_scale_factor)

            x_cand = np.arange(1, width + 1)
            # find the maximum of the multiple of resampling_scale_factor
            resample_width_end = np.max(x_cand[x_cand %
                                               resampling_scale_factor == 0])

            # resample SLC
            resampled_slc = resample(
                slc_demodulate[:, :resample_width_end], sub_width, axis=1)

            meta['range_spacing'] = \
                self.rg_pxl_spacing * resampling_scale_factor
            meta['slant_range'] = \
                self.slant_range(0) + \
                np.arange(sub_width) * meta['range_spacing']

            return resampled_slc, meta

        else:
            meta['range_spacing'] = self.rg_pxl_spacing
            meta['slant_range'] = \
                self.slant_range(0) + \
                np.arange(width) * meta['range_spacing']

            return slc_demodulate, meta

    def bandpass_spectrum(self,
                          slc_raster,
                          low_frequency,
                          high_frequency,
                          window_function,
                          window_shape=0.25,
                          fft_size=None,
                          ):
        """Bandpass SLC for given center frequency and bandwidth

        Parameters
        ----------
        slc_raster : numpy.ndarray
            numpy array of slc raster,
        low_frequency : float
            low  frequency of band to be passed [Hz]
        high_frequency : float
            high frequency band to be passed [Hz]
        window_function: str
            window type {'tukey', 'kaiser', 'cosine'}
        window_shape : float
            parameter for the window shape
            kaiser 0<= window_shape < inf
            tukey and cosine 0 <= window_shape <= 1
        fft_size : int
            fft size.

        Returns
        -------
        slc_bandpassed : numpy.ndarray
            numpy array of bandpassed slc
        """
        error_channel = journal.error('splitspectrum.bandpass_spectrum')

        rg_sample_freq = self.rg_sample_freq
        rg_bandwidth = self.rg_bandwidth
        center_frequency = self.center_frequency
        height, width = slc_raster.shape
        slc_raster = np.asanyarray(slc_raster, dtype='complex')
        new_bandwidth = high_frequency - low_frequency
        new_rg_sample_freq = self.sampling_bandwidth_ratio * new_bandwidth
        resampling_scale_factor = rg_sample_freq / new_rg_sample_freq

        if new_bandwidth < 0:
            err_str = "Low frequency is higher than high frequency"
            error_channel.log(err_str)
            raise ValueError(err_str)

        if fft_size is None:
            fft_size = width

        if fft_size < width:
            err_str = "FFT size is smaller than number of range bins"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # construct window to be deconvolved
        # from the original SLC in freq domain
        window_target = self.get_range_bandpass_window(
            center_frequency=0,
            freq_low=-rg_bandwidth/2,
            freq_high=rg_bandwidth/2,
            sampling_frequency=rg_sample_freq,
            fft_size=fft_size,
            window_function=window_function,
            window_shape=window_shape
            )
        # construct window to bandpass spectrum
        # for given low and high frequencies
        window_bandpass = self.get_range_bandpass_window(
            center_frequency=0,
            freq_low=low_frequency - center_frequency,
            freq_high=high_frequency - center_frequency,
            sampling_frequency=rg_sample_freq,
            fft_size=fft_size,
            window_function=window_function,
            window_shape=window_shape
            )

        # remove the windowing effect from the spectrum
        spectrum_target = fft(slc_raster, n=fft_size, workers=-1)
        spectrum_target = np.divide(spectrum_target,
                                    window_target,
                                    out=np.zeros_like(spectrum_target),
                                    where=window_target != 0)

        # apply new bandpass window to spectrum
        slc_bandpassed = ifft(spectrum_target
                              * window_bandpass
                              * np.sqrt(resampling_scale_factor),
                              n=fft_size,
                              workers=-1)

        return slc_bandpassed

    def demodulate_slc(self, slc_array, diff_frequency, rg_sample_freq):
        """ Demodulate SLC

        If diff_frequency is not zero, then the spectrum of SLC is shifted
        so that the sub-band slc is demodulated to center the sub band spectrum

        Parameters
        ----------
        slc_array : numpy.ndarray
            SLC raster or block of SLC raster
        diff_frequency : float
            difference between original and new center frequency [Hz]
        rg_sample_freq : float
            range sampling frequency [Hz]

        Returns
        -------
        slc_baseband  : numpy.ndarray
            demodulated SLC
        """
        height, width = slc_array.shape
        range_time = np.arange(width) / rg_sample_freq
        slc_shifted = slc_array * np.exp(1j * 2.0 * np.pi
                                         * diff_frequency * range_time)
        return slc_shifted

    def freq_spectrum(self, cfrequency, dt, fft_size):
        ''' Return Discrete Fourier Transform sample frequencies
        with center frequency bias.

        Parameters:
        ----------
        cfrequency : float
            Center frequency (Hz)
        dt : float
            Sample spacing.
        fft_size : int
            Window length.
        Returns:
        -------
        freq : ndarray
            Array of length fft_size containing sample frequencies.
        '''
        freq = cfrequency + fftfreq(fft_size, dt)
        return freq

    def get_range_bandpass_window(self,
                                  center_frequency,
                                  sampling_frequency,
                                  fft_size,
                                  freq_low,
                                  freq_high,
                                  window_function='tukey',
                                  window_shape=0.25):
        '''Get range bandpass window such as Tukey, Kaiser, cosine.
        Window is constructed in frequency domain from low to high frequencies
        with given window_function and shape.

        Parameters
        ----------
        center_frequency : float
            Center frequency of frequency bin [Hz]
            If slc is basebanded, center_frequency can be 0.
        sampling_frequency : float
            sampling frequency [Hz]
        fft_size : int
            fft size
        freq_low : float
            low frequency to be passed [Hz]
        freq_high: float
            high frequency to be passed [Hz]
        window_function : str
            window type {tukey, kaiser, cosine}
        window_shape : float
            parameter for the window shape
            kaiser 0<= window_shape < inf
            tukey and cosine 0 <= window_shape <= 1

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional bandpass filter in frequency domain
        '''
        error_channel = journal.error('splitspectrum.get_range_bandpass_window')
        # construct the frequency bin [Hz]
        frequency = self.freq_spectrum(
                    cfrequency=center_frequency,
                    dt=1.0/sampling_frequency,
                    fft_size=fft_size
                    )

        window_kind = window_function.lower()

        # Windowing effect will appear from freq_low to freq_high 
        # for given frequency bin
        if window_kind == 'tukey':
            if not (0 <= window_shape <= 1):
                err_str = f"Expected window_shape between 0 and 1, got {window_shape}."
                error_channel.log(err_str)
                raise ValueError(err_str)

            filter_1d = self.construct_range_bandpass_tukey(
                frequency_range=frequency,
                freq_low=freq_low,
                freq_high=freq_high,
                window_shape=window_shape
            )

        elif window_kind == 'kaiser':
            if not (window_shape > 0):
                err_str = f"Expected pedestal bigger than 0, got {window_shape}."
                error_channel.log(err_str)
                raise ValueError(err_str)

            filter_1d = self.construct_range_bandpass_kaiser(
                frequency_range=frequency,
                freq_low=freq_low,
                freq_high=freq_high,
                window_shape=window_shape
            )

        elif window_kind == 'cosine':
            if not (0 <= window_shape <= 1):
                err_str = f"Expected window_shape between 0 and 1, got {window_shape}."
                error_channel.log(err_str)
                raise ValueError(err_str)
            filter_1d = self.construct_range_bandpass_cosine(
                frequency_range=frequency,
                freq_low=freq_low,
                freq_high=freq_high,
                window_shape=window_shape
            )

        else:
            err_str = f"window {window_kind} not in (Kaiser, Cosine, Tukey)."
            error_channel.log(err_str)
            raise ValueError(err_str)

        return filter_1d

    def construct_range_bandpass_cosine(self,
                                        frequency_range,
                                        freq_low,
                                        freq_high,
                                        window_shape):
        '''Generate a Cosine bandpass window

        Parameters
        ----------
        frequency_range : np.ndarray
            Discrete Fourier Transform sample frequency range bins[Hz]
        freq_low : float
            low frequency to be passed [Hz]
        freq_high: float
            high frequency to be passed [Hz]
        window_shape : float
            parameter for the cosine window

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional Cosine bandpass filter in frequency domain
        '''
        filter_1d = self._construct_range_bandpass_kaiser_cosine(
            frequency_range,
            freq_low,
            freq_high,
            cosine_window,
            window_shape)
        return filter_1d

    def construct_range_bandpass_kaiser(self,
                                        frequency_range,
                                        freq_low,
                                        freq_high,
                                        window_shape):
        '''Generate a Kaiser bandpass window

        Parameters
        ----------
        frequency_range : np.ndarray
            Discrete Fourier Transform sample frequency range bins[Hz]
        freq_low : float
            low frequency to be passed [Hz]
        freq_high: float
            high frequency to be passed [Hz]
        window_shape : float
            parameter for the kaiser window

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional kaiser bandpass filter in frequency domain
        '''
        filter_1d = self._construct_range_bandpass_kaiser_cosine(
            frequency_range,
            freq_low,
            freq_high,
            np.kaiser,
            window_shape)
        return filter_1d

    def _construct_range_bandpass_kaiser_cosine(
            self,
            frequency_range,
            freq_low,
            freq_high,
            window_function,
            window_shape):
        '''Generate a Kaiser or cosine bandpass window

        Parameters
        ----------
        frequency_range : np.ndarray
            Discrete Fourier Transform sample frequency range bins[Hz]
        freq_low : float
            low frequency to be passed [Hz]
        freq_high: float
            high frequency to be passed [Hz]
        window_function : class function
            window type {np.kaiser, cosine_window}
        window_shape : float
            parameter for the kaiser window

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional kaiser bandpass filter in frequency domain
        '''
        error_channel = journal.error(
            'splitspectrum._construct_range_bandpass_kaiser_cosine')

        subbandwidth = np.abs(freq_high - freq_low)
        fft_size = len(frequency_range)

        if freq_high > np.max(frequency_range):
            err_str = "High frequency is out of frequency bins."
            error_channel.log(err_str)
            raise ValueError(err_str)

        if freq_low < np.min(frequency_range):
            err_str = "Low frequency is out of frequency bins."
            error_channel.log(err_str)
            raise ValueError(err_str)

        # sampling frequency is 1.2 times wider than bandwidth for NISAR
        sampling_bandwidth_ratio = self.sampling_bandwidth_ratio

        sampling_low_frequency = \
            freq_low - (sampling_bandwidth_ratio - 1) * subbandwidth * 0.5
        sampling_high_frequency = \
            freq_high + (sampling_bandwidth_ratio - 1) * subbandwidth * 0.5

        # index for low and high sampling frequency in frequency_range
        idx_freq_low = np.abs(
            frequency_range - sampling_low_frequency).argmin()
        idx_freq_high = np.abs(
            frequency_range - sampling_high_frequency).argmin()

        if idx_freq_low >= idx_freq_high:
            subband_length = idx_freq_high + fft_size - idx_freq_low + 1
        else:
            subband_length = idx_freq_high - idx_freq_low + 1

        filter_1d = np.zeros([fft_size], dtype='complex')

        # window_function is function class {np.kaiser or consine}
        subwindow = window_function(subband_length, window_shape)

        if idx_freq_low >= idx_freq_high:
            filter_1d[idx_freq_low:] = subwindow[0:fft_size - idx_freq_low]
            filter_1d[: idx_freq_high + 1] = subwindow[fft_size - idx_freq_low:]
        else:
            filter_1d[idx_freq_low:idx_freq_high+1] = subwindow

        return filter_1d

    def construct_range_bandpass_tukey(self,
                                       frequency_range,
                                       freq_low,
                                       freq_high,
                                       window_shape):
        '''Generate a Tukey (raised-cosine) window

        Parameters
        ----------
        frequency_range : np.ndarray
            Discrete Fourier Transform sample frequency range [Hz]
        freq_low : float
            low frequency to be passed [Hz]
        freq_high, : list of float
            high frequency to be passed [Hz]
        window_shape : float
            parameter for the Tukey (raised cosine) filter

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional Tukey bandpass filter in frequency domain
        '''

        fft_size = len(frequency_range)
        freq_mid = 0.5 * (freq_low + freq_high)
        subbandwidth = np.abs(freq_high - freq_low)
        df = 0.5 * subbandwidth * window_shape

        filter_1d = np.zeros(fft_size, dtype='complex')
        for i in range(0, fft_size):
            # Get the absolute value of shifted frequency
            freq = frequency_range[i]
            freqabs = np.abs(freq - freq_mid)
            # Passband. i.e. range of frequencies that can pass
            # through a filter
            if (freq <= (freq_high - df)) and (freq >= (freq_low + df)):
                filter_1d[i] = 1
            # Transition region
            elif ((freq < freq_low + df) and (freq >= freq_low - df)) \
                    or ((freq <= freq_high + df) and (freq > freq_high - df)):
                filter_1d[i] = 0.5 * (
                    1.0 + np.cos(np.pi / (subbandwidth * window_shape)
                    * (freqabs - 0.5 * (1.0 - window_shape) * subbandwidth)))
        return filter_1d
