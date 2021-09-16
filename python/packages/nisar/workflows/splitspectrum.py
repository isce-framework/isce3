import numpy as np
import journal
from scipy.fft import fft, ifft
from scipy.signal import resample
from numpy.fft import fftfreq

import isce3


def get_meta_data_bandpass(slc_product, freq):
    """get meta data from SLC object 
    Parameters
    ----------
    slc_product : slc object
    freq : frequency A or B

    Returns
    -------
    meta_data : dict containing meta_data
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


def check_insar_mixmode(ref_slc, sec_slc, pols):
    """check if bandpass is needed
    if necessary, determine which SLC will be bandpassed
    Parameters
    ----------
    ref_slc : reference SLC object
    sec_slc : secondar SLCobject
    pol : polarization list 

    Returns
    -------
    mode : dict for slc to be bandpassed
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

    def __init__(self, rslc_product, freq):
        self.rslc_product = rslc_product
        self.freq = freq
        rdr_grid = self.rslc_product.getRadarGrid(self.freq)
        self.rg_pxl_spacing = rdr_grid.range_pixel_spacing
        self.wavelength = rdr_grid.wavelength
        self.rg_sample_freq = isce3.core.speed_of_light / 2.0 / self.rg_pxl_spacing
        self.rg_bandwidth = self.rslc_product.getSwathMetadata(
            self.freq).processed_range_bandwidth
        self.center_frequency = isce3.core.speed_of_light / self.wavelength
        self.slant_range = rdr_grid.slant_range
        self.beta = 0.25
        
    def bandpass_spectrum(self, 
                          slc_raster, 
                          low_frequency, 
                          high_frequency,
                          new_center_frequency,
                          beta=None, 
                          fft_size=None):
        """bandpass range specturm for given center frequency and bandwidth
        Parameters
        ----------
        slc_raster : numpy array of slc 
        low_frequency : low  frequency for bandpass [Hz]
        high_frequency : high frequency for bandpass [Hz]
        new_center_frequency : new center frequency for bandpass [Hz]
        beta : parameter for the raised cosine filter
        fft_size : integer fft size 
        Returns
        -------
        filtered_slc : numpy array of bandpassed slc
        meta : dict for meta data
        """       
        error_channel = journal.error('splitspectrum.bandpass_spectrum')
 
        rg_sample_freq = self.rg_sample_freq
        rg_bandwidth = self.rg_bandwidth
        center_frequency = self.center_frequency
        diff_frequency = self.center_frequency - new_center_frequency
        height, width = slc_raster.shape
        slc_raster = np.array(slc_raster, dtype='complex')
        new_bandwidth = high_frequency - low_frequency
        
        if new_bandwidth < 0:
            err_str = f"Low and high frequencies values are wrong"
            error_channel.log(err_str)
            raise ValueError(err_str)

        if fft_size is None:
            fft_size = width
        if beta is None:
            beta = self.beta

        window_target = self.construct_range_bandpass_cosine(
            suband_center_frequencies=0,
            frequencyLH=[-rg_bandwidth/2, 
                          rg_bandwidth/2],
            sampling_frequency=rg_sample_freq, 
            fft_size=fft_size,
            beta=beta
        )

        bpfilter = self.construct_range_bandpass_cosine(
            suband_center_frequencies=center_frequency,
            frequencyLH=[low_frequency, high_frequency],
            sampling_frequency=rg_sample_freq, 
            fft_size=fft_size,
            beta=beta
        )

        resampling_scale_factor = rg_bandwidth / int(np.abs(new_bandwidth))
        sub_fft_size = int(fft_size/resampling_scale_factor)

        # remove the windowing effect from the spectrum 
        spectrum_target = fft(np.array(slc_raster)) / \
                              np.tile(window_target, [height, 1])
        # apply new bandpass window to spectrum 
        slc_bp = ifft(spectrum_target 
                      * np.tile(bpfilter, [height, 1]) 
                      * np.sqrt(resampling_scale_factor))      

        # demodulate the SLC to be baseband to new center frequency
        slc_demodulate = self.demodulate_slc(slc_bp, 
                                             diff_frequency, 
                                             rg_sample_freq)

        # resample SLC 
        filtered_slc = resample(slc_demodulate, sub_fft_size, axis=1)

        meta = dict()
        meta['center_frequency'] = new_center_frequency
        meta['rg_bandwidth'] = new_bandwidth
        meta['range_spacing'] = self.rg_pxl_spacing * resampling_scale_factor
        meta['slant_range'] = np.linspace(self.slant_range(0), self.slant_range(fft_size),\
                            sub_fft_size, endpoint=False)

        return filtered_slc, meta

    def demodulate_slc(self, slc_array, diff_frequency, rg_sample_freq):
        """ demodulate slc 

        Parameters
        ----------
        slc_array : slc numpy array
        diff_frequency : shift frequency [Hz] 
        rg_sample_freq : range sampling frequency [Hz]
        Returns
        -------
        slc_de  : demodulated slc
        """
        height, width = np.shape(slc_array)

        range_time=np.zeros(width)
        for ii in range(width):
            range_time[ii] = ii / rg_sample_freq
        
        slc_de = np.zeros([height, width], dtype=complex)
        for jj in range(height):
            slc_de[jj, :] = slc_array[jj,:] * np.exp(-1 * 1j * 2.0 * np.pi * -1 
                                                    * diff_frequency * range_time)
        return slc_de

    def freq_spectrum(self, cfrequency, dt, fft_size):
        freq = cfrequency + fftfreq(fft_size, dt)
        return freq

    def construct_range_bandpass_cosine(self, 
                                        suband_center_frequencies,
                                        sampling_frequency,
                                        fft_size,
                                        frequencyLH,
                                        beta):
        '''Turkey window
        Parameters
        ----------
        suband_center_frequencies : center frequency for passband [Hz]
        sampling_frequency :  sampling frequency [Hz]
        fft_size : low  frequency for bandpass [Hz]
        frequencyLH : vector of low and high frequency [Hz]
        beta : parameter for the raised cosine filter

        Returns
        -------
        filter_1d : one dimensional cosine bandpass filter in frequency domain
        '''
        frequency = self.freq_spectrum(
                        cfrequency = suband_center_frequencies,
                        dt = 1.0 / sampling_frequency,
                        fft_size = fft_size
                        ) 

        freq_size = np.shape(frequency)[0]
        norm = 1.0 
        fL, fH = frequencyLH
        freq_mid = 0.5 * (fH+fL)
        subbandwidth = np.abs(fH-fL)
        df = 0.5 * subbandwidth * beta

        filter_1d = np.zeros([freq_size], dtype='complex')
        for i in range(0, freq_size):

            # Get the absolute value of shifted frequency
            freq = np.abs(frequency[i])
            freqabs = np.abs(freq - freq_mid)

            # Passband
            if (freq <= (fH - df)) and (freq >= (fL + df)):
                filter_1d[i] = 1

            # Transition region
            elif ((freq < (fL + df)) and (freq >= (fL - df))) \
                    or ((freq <= (fH + df)) and (freq > (fH - df))):
                filter_1d[i] = norm * 0.5 * (1.0 
                             + np.cos(np.pi / (subbandwidth*beta) 
                             * (freqabs - 0.5 * (1.0 - beta) * subbandwidth)))
        return filter_1d
