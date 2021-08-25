#!/usr/bin/env python3

import numpy as np
from scipy.fft import fft, ifft, fftshift
from scipy.signal import kaiserord, freqz, firwin, resample
from numpy.fft import fftfreq
import isce3

def meta_data_bandpass(frame, freq):
    meta_data = dict()
    rdr_grid         = frame.getRadarGrid(freq)
    meta_data['rg_pxl_spacing']   = rdr_grid.range_pixel_spacing
    meta_data['wavelength']      = rdr_grid.wavelength
    meta_data['rg_sample_freq']   = isce3.core.speed_of_light * 0.5 / meta_data['rg_pxl_spacing']
    meta_data['rg_bandwidth']     = frame.getSwathMetadata(freq).processed_range_bandwidth
    meta_data['center_frequency']  = isce3.core.speed_of_light / meta_data['wavelength'] 
    meta_data['slant_range']      = rdr_grid.slant_range
    return meta_data

def insar_mixmode(ref_slc, sec_slc, pols):
    """
    insar_mixmode(ref_slc, sec_slc, pols)
    check if bandpass is necessary
    if necessary, determine SLC to be bandpassed
    ref_slc : reference slc object
    sec_slc : secondary slc object
    pol     : polarization list [frequency, polarization]
    """
    mode = dict()

    for freq, pol_list in pols.items():
        ref_meta_data = meta_data_bandpass(ref_slc, freq)
        sec_meta_data = meta_data_bandpass(sec_slc, freq)

        # check if two SLCs have same bandwidth and center frequency
        if (ref_meta_data['wavelength'] != sec_meta_data['wavelength'] ) \
        or (ref_meta_data['rg_bandwidth'] != sec_meta_data['rg_bandwidth']):

            if ref_meta_data['rg_bandwidth'] > sec_meta_data['rg_bandwidth']:
                mode[freq] = 'ref'
            else:
                mode[freq] = 'sec'
    return mode


class Splitspectrum:
    
    def __init__(self):
        self.frame = ""
        self.slc_raster = 0
        self.fft_size = 0
        self.width = 0
        self.low_frequency = 0
        self.high_frequency = 0
        self.new_center_frequency = 0
        self.freq = 'A'
        self.pol = 'HH'
        self.beta = 0.25
        
        return None
      
    def bandpass_spectrum(self):
        """
        Band-pass filter
        
        frame : slc oject to contain parameters
        slc   : numpy array of slc 
        fL    : low  frequency for bandpass
        fH    : high frequency for bandpass
        cfrequency : new center frequency for bandpass
        freq : [A , B]
        """
        rdr_grid         = self.frame.getRadarGrid(self.freq)
        rg_pxl_spacing   = rdr_grid.range_pixel_spacing
        wavelength       = rdr_grid.wavelength
        rg_sample_freq   = isce3.core.speed_of_light / 2.0 / rg_pxl_spacing
        rg_bandwidth     = self.frame.getSwathMetadata(self.freq).processed_range_bandwidth
        center_frequency = isce3.core.speed_of_light / wavelength
        slant_range      = rdr_grid.slant_range
        
        fL = self.low_frequency
        fH = self.high_frequency
        
        diff_frequency = center_frequency - self.new_center_frequency
        height, width = self.slc_raster.shape
        slc_raster = np.array(self.slc_raster,dtype='complex')
        
        if self.fft_size == 0 : fft_size = width

        window_target = self.constructRangeBandpassCosine(
                        suband_center_frequencies = 0,
                        frequencyLH = [- rg_bandwidth/2, 
                                        rg_bandwidth/2],
                        samplingFrequency = rg_sample_freq, 
                        fft_size = fft_size,
                        beta = 0.25)

        bpfilter = self.constructRangeBandpassCosine(
                        suband_center_frequencies = center_frequency,
                        frequencyLH = [fL, fH],
                        samplingFrequency = rg_sample_freq, 
                        fft_size = fft_size,
                        beta = 0.25)
        
        resampling_factor = rg_bandwidth / int(np.abs(fH-fL))
        sub_fft_size = int(fft_size/resampling_factor)
        power_factor = resampling_factor /1.1

        # remove the windowing effect from the spectrum 
        spectrum_target = fft(np.array(slc_raster)) / np.tile(window_target,[height, 1])
        # apply new bandpass window to spectrum 
        slc_bp = ifft(spectrum_target * np.tile(bpfilter,[height, 1])*np.sqrt(power_factor))        
        # demodulate the SLC to be baseband to new center frequency
        slc_demodulate = self.demodulate_slc(slc_bp, diff_frequency, rg_sample_freq)

        # resample SLC 
        filtered_slc = resample(slc_demodulate, sub_fft_size ,axis=1)

        meta = dict()
        meta['centerFrequency'] = self.new_center_frequency
        meta['Bandwidth'] = fH - fL
        meta['rangeSpacing'] = rg_pxl_spacing * resampling_factor
        meta['slantRange'] =  np.linspace(slant_range(0),slant_range(fft_size),\
                            sub_fft_size,endpoint=False)

        return filtered_slc, meta

    def demodulate_slc(self, 
                       slc_array, diff_frequency, rg_sample_freq):
        """
        demodulate slc_array for a given frequecy diff_frequency
        
        slc   : numpy array of SLC 
        diff_frequency : frequency difference between old and new slcs
        rg_sample_freq : range sampling frequency       
        """
        height, width = np.shape(slc_array)

        rangeTime=np.zeros(width)
        for ii in range(width):
            rangeTime[ii] = ii / rg_sample_freq
        
        slc_de = np.zeros([height,width],dtype=complex)
        for jj in range(height):
            slc_de[jj,:] = slc_array[jj,:] * np.exp(-1 * 1j * 2.0 * np.pi * -1 * \
                            diff_frequency * rangeTime)
        return slc_de

    def freqSpectrum(self, cfrequency, dt, fft_size):
        freq = cfrequency + fftfreq(fft_size, dt)
        return freq

    def constructRangeBandpassCosine(self, suband_center_frequencies,
                                    samplingFrequency,
                                    fft_size,
                                    frequencyLH,
                                    beta):
       
        frequency = self.freqSpectrum(
                    cfrequency = suband_center_frequencies,
                    dt = 1.0 / samplingFrequency,
                    fft_size = fft_size) 

        freq_size=np.shape(frequency)[0]
        norm = 1.0;    
        fL, fH = frequencyLH
        fmid = (fH + fL) * 0.5 
        subbandwidth = np.abs( fH - fL )
        df = 0.5 * subbandwidth * beta

        filter_1d=np.zeros([freq_size],dtype='complex')
        for i in range(0,freq_size):

            #Get the absolute value of shifted frequency
            freq = np.abs(frequency[i])
            freqabs = np.abs(freq- fmid)

            # Passband
            if (freq <= (fH - df)) and (freq >= (fL + df)):
                filter_1d[i] = 1

            # Transition region
            elif ((freq < (fL + df)) and (freq >= (fL - df))) \
            or ((freq <= (fH + df)) and (freq > (fH - df))):
                filter_1d[i] = norm * 0.5 * (1.0 + np.cos(np.pi / (subbandwidth*beta) *
                            (freqabs - 0.5 * (1.0 - beta) * subbandwidth)))
        return filter_1d
