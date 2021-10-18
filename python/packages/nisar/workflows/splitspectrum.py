import numpy as np
import journal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import resample

import isce3
from nisar.workflows.focus import cosine_window


def get_meta_data_bandpass(slc_product, freq):
    """Get meta data from SLC object.

    Parameters
    ----------
    slc_product : nisar.products.readers.SLC
        slc object
    freq : {'A', 'B'}
        frequency band

    Returns
    -------
    meta_data : dict
        dict containing meta_data
    """
    meta_data = dict()
    rdr_grid = slc_product.getRadarGrid(freq)
    meta_data['rg_pxl_spacing'] = rdr_grid.range_pixel_spacing
    meta_data['wavelength'] = rdr_grid.wavelength
    meta_data['rg_sample_freq'] = isce3.core.speed_of_light * \
        0.5 / meta_data['rg_pxl_spacing']
    meta_data['rg_bandwidth'] = slc_product.getSwathMetadata(
        freq).processed_range_bandwidth
    meta_data['center_frequency'] = isce3.core.speed_of_light / \
        meta_data['wavelength']
    meta_data['slant_range'] = rdr_grid.slant_range
    return meta_data


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
    pol : str 
        String indicating the polarization 

    Returns
    -------
    mode : dict 
        Dict mapping frequency band (e.g. "A" or "B") to 
        SLC to be bandpassed ("ref" or "sec").
    """
    mode = dict()

    for freq, pol_list in pols.items():
        ref_meta_data = get_meta_data_bandpass(ref_slc, freq)
        sec_meta_data = get_meta_data_bandpass(sec_slc, freq)
        
        ref_wvl = ref_meta_data['wavelength']
        sec_wvl = sec_meta_data['wavelength']
        ref_bw = ref_meta_data['rg_bandwidth']
        sec_bw = sec_meta_data['rg_bandwidth']

        # check if two SLCs have same bandwidth and center frequency
        if (ref_wvl != sec_wvl) or (ref_bw != sec_bw):
            if ref_bw > sec_bw:
                mode[freq] = 'ref'
            else:
                mode[freq] = 'sec'
    return mode


class SplitSpectrum:
    '''
    Split the range spectrum in InSAR workflow
    '''

    def __init__(self, 
                 rg_sample_freq, 
                 rg_bandwidth, 
                 center_frequency, 
                 slant_range,  
                 freq):
        """Initialized Bandpass Class with SLC meta data
        
        Parameters
        ----------
        rg_sample_freq : float
            range sampling freqeuncy 
        rg_bandwidth : float
            range bandwidth [Hz]
        center_frequency : float 
            center frequency of SLC [Hz]
        slant_range : new center frequency for bandpass [Hz]
        freq : {'A', 'B'}
            frequency band
        """       
        self.freq = freq
        self.rg_sample_freq = rg_sample_freq
        self.rg_pxl_spacing = isce3.core.speed_of_light / 2.0 / self.rg_sample_freq
        self.rg_bandwidth = rg_bandwidth
        self.center_frequency = center_frequency
        self.slant_range = slant_range
        
    def bandpass_spectrum(self, 
                          slc_raster, 
                          low_frequency, 
                          high_frequency,
                          new_center_frequency,
                          window,
                          window_shape=0.25, 
                          fft_size=None
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
        new_center_frequency : float
            new center frequency for bandpass [Hz]
        window_shape : float 
            parameter for the raised cosine filter (e.g. 0 ~ 1)
        fft_size : int 
            fft size. 

        Returns
        -------
        filtered_slc : numpy.ndarray 
            numpy array of bandpassed slc
        meta : dict 
            dict containing meta data of bandpassed slc
            center_frequency, rg_bandwidth, range_spacing, slant_range
        """       
        error_channel = journal.error('splitspectrum.bandpass_spectrum')

        rg_sample_freq = self.rg_sample_freq
        rg_bandwidth = self.rg_bandwidth
        center_frequency = self.center_frequency
        diff_frequency = self.center_frequency - new_center_frequency
        height, width = slc_raster.shape
        slc_raster = np.asanyarray(slc_raster, dtype='complex')
        new_bandwidth = high_frequency - low_frequency
        
        if new_bandwidth < 0:
            err_str = f"Low frequency is higher than high frequency"
            error_channel.log(err_str)
            raise ValueError(err_str)

        if fft_size is None:
            fft_size = width
        
        if fft_size < width:
            err_str = f"FFT size is smaller than range bins"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # construct window to remove window effect in freq domain
        window_target = self.get_range_bandpass_window(
            center_frequency=0,
            frequencyLH=[-rg_bandwidth/2, 
                          rg_bandwidth/2],
            sampling_frequency=rg_sample_freq, 
            fft_size=fft_size,
            window_function=window,
            window_shape=window_shape
        )
        print(window, window_shape)
        # construct window to bandpass spectrum 
        # for given low and high frequencies
        window_bandpass = self.get_range_bandpass_window(
            center_frequency=0,
            frequencyLH=[low_frequency - center_frequency, 
                         high_frequency - center_frequency],
            sampling_frequency=rg_sample_freq, 
            fft_size=fft_size,
            window_function=window,
            window_shape=window_shape
        )

        resampling_scale_factor = rg_bandwidth / new_bandwidth
        sub_fft_size = int(width / resampling_scale_factor)

        # remove the windowing effect from the spectrum 
        spectrum_target = fft(slc_raster, n=fft_size) / \
                              window_target
        # apply new bandpass window to spectrum 
        slc_bp = ifft(spectrum_target 
                      * window_bandpass
                      * np.sqrt(resampling_scale_factor), n=fft_size)

        # demodulate the SLC to be baseband to new center frequency
        # if fft_size > width, then crop the spectrum from 0 to width
        slc_demodulate = self.demodulate_slc(slc_bp[:, :width], 
                                             diff_frequency, 
                                             rg_sample_freq)

        # resample SLC 
        filtered_slc = resample(slc_demodulate, sub_fft_size, axis=1)

        meta = dict()
        meta['center_frequency'] = new_center_frequency
        meta['rg_bandwidth'] = new_bandwidth
        meta['range_spacing'] = self.rg_pxl_spacing * resampling_scale_factor
        meta['slant_range'] = np.linspace(self.slant_range(0), self.slant_range(width),\
                            sub_fft_size, endpoint=False)

        return filtered_slc, meta

    def demodulate_slc(self, slc_array, diff_frequency, rg_sample_freq):
        """ Demodulate SLC 
        
        If diff_frequency is not zero, then the spectrum of SLC is shifted 
        so that the subband slc is demodulated to center the sub band spectrum

        Parameters
        ----------
        slc_array : numpy.ndarray 
            SLC raster or blocked SLC raster
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
        slc_baseband = slc_array * np.exp(-1 * 1j * 2.0 * np.pi * -1 
                                     * diff_frequency * range_time)
        return slc_baseband

    def freq_spectrum(self, cfrequency, dt, fft_size):
        freq = cfrequency + fftfreq(fft_size, dt)
        return freq

    def get_range_bandpass_window(self, 
                                  center_frequency,
                                  sampling_frequency,
                                  fft_size,
                                  frequencyLH,
                                  window_function='tukey',
                                  window_shape=0.25):
        '''Get range bandpas window {tukey, kaiser, cosine}
        Window is constructed in frequency domain from low to high frequencies (frequencyLH)
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
        frequencyLH : list of float
            vector of low and high frequency [Hz]
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
        #
        frequency = self.freq_spectrum(
                    cfrequency=center_frequency,
                    dt=1.0/sampling_frequency,
                    fft_size=fft_size
                    ) 
        
        fL, fH = frequencyLH   
        window_kind = window_function.lower()
        
        if window_kind == 'tukey':
            if not (0 <= window_shape <= 1):
                raise ValueError(f"Expected window_shape between 0 and 1, got {window_shape}.")

            filter_1d = self.construct_range_bandpass_tukey(
                frequency_range=frequency,
                frequencyLH=[fL, fH],
                window_shape=window_shape
            )
        
        elif window_kind == 'kaiser' or window_kind == 'cosine':
            
            if (window_kind == 'kaiser') and not (window_shape > 0):
                raise ValueError(f"Expected pedestal bigger than 0, got {window_shape}.")
                
            if (window_kind == 'cosine') and not (0 <= window_shape <= 1):
                raise ValueError(f"Expected window_shape between 0 and 1, got {window_shape}.")
                    
            filter_1d = self.construct_range_bandpass_kaiser_cosine(
                frequency_range=frequency,
                frequencyLH=[fL, fH],
                rg_sample_freq=sampling_frequency,
                window_function=window_kind,
                window_shape=window_shape
            )
        
        else:
            raise NotImplementedError(f"window {window_kind} not in (Kaiser, Cosine, Tukey).")

        return filter_1d

    def construct_range_bandpass_kaiser_cosine(self, 
                                               frequency_range,
                                               frequencyLH,
                                               rg_sample_freq,
                                               window_function,
                                               window_shape):
        '''Generate a kaiser window

        Parameters
        ----------
        frequency_range : np.ndarray
            Discrete Fourier Transform sample frequency range bins[Hz]
        frequencyLH : list of float 
            list of low and high frequency [Hz]
        window_shape : float 
            parameter for the kaiser window

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional kaiser bandpass filter in frequency domain
        '''

        fL, fH = frequencyLH
        subbandwidth = np.abs(fH - fL)
        fft_size = len(frequency_range)
        
        assert fH < np.max(frequency_range), 'High freqeuncy is expected '
        assert fL > np.min(frequency_range), 'Low freqeuncy is lower than observation'
       
        # sampling frequency is 1.2 times wider than bandwith
        sampling_bandwidth_ratio = 1.2
        sampling_low_frequency = fL - (sampling_bandwidth_ratio - 1) * subbandwidth *.5
        sampling_high_frequency = fH + (sampling_bandwidth_ratio - 1) * subbandwidth *.5

        # index for low and high sampling frequency in frequency_range
        idx_fL = np.abs(frequency_range - sampling_low_frequency).argmin()
        idx_fH = np.abs(frequency_range - sampling_high_frequency).argmin()
        
        if idx_fL >= idx_fH: 
            subband_length = idx_fH + fft_size - idx_fL
        else:
            subband_length = idx_fH - idx_fL
            
        filter_1d = np.zeros([fft_size], dtype='complex')
        if window_function == 'kaiser':
            subwindow = np.kaiser(subband_length, window_shape)
        elif window_function == 'cosine':
            subwindow = cosine_window(subband_length, window_shape)
        
        if idx_fL >= idx_fH: 
            filter_1d[idx_fL :] = subwindow[0 : (fft_size - idx_fL)]
            filter_1d[: idx_fH] = subwindow[(fft_size - idx_fL):]
        else:
            filter_1d[idx_fL : idx_fH] = subwindow
            
        return filter_1d

    def construct_range_bandpass_tukey(self,
                                       frequency_range,
                                       frequencyLH,
                                       window_shape):
        '''Generate a Tukey (raised-cosine) window

        Parameters
        ----------
        frequency_range : np.ndarray
            Discrete Fourier Transform sample frequency range [Hz]
        frequencyLH : list of float 
            vector of low and high frequency [Hz]
        window_shape : float 
            parameter for the tukey (raised cosine) filter

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional cosine bandpass filter in frequency domain
        '''
        fL, fH = frequencyLH
        subbandwidth = np.abs(fH - fL)
        fft_size = len(frequency_range)
        
        norm = 1.0 
        fL, fH = frequencyLH
        freq_mid = 0.5 * (fH + fL)
        subbandwidth = np.abs(fH - fL)
        df = 0.5 * subbandwidth * window_shape

        filter_1d = np.zeros([fft_size], dtype='complex')
        for i in range(0, fft_size):
            # Get the absolute value of shifted frequency
            freq = frequency_range[i]
            freqabs = np.abs(freq - freq_mid)
            # Passband
            if (freq <= (fH - df)) and (freq >= (fL + df)):
                filter_1d[i] = 1
            # Transition region
            elif ((freq < (fL + df)) and (freq >= (fL - df))) \
                    or ((freq <= (fH + df)) and (freq > (fH - df))):
                filter_1d[i] = norm * 0.5 * (1.0 
                            + np.cos(np.pi / (subbandwidth * window_shape) 
                            * (freqabs - 0.5 * (1.0 - window_shape) * subbandwidth)))
        return filter_1d



