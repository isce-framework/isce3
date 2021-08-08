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
    meta_data['center_frequecny']  = isce3.core.speed_of_light / meta_data['wavelength'] 
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

def bandpass_spectrum(frame,
                       slc, 
                       fL, 
                       fH, 
                       cfrequency,
                       freq, 
                       block_rows):
    """
    Band-pass filter
    
    frame : SLC oject to contain parameters
    slc   : numpy array of SLC 
    fL    : Low  frequency for bandpass
    fH    : High frequency for bandpass
    cfrequency : new center frequency for bandpass
    freq : [A , B]
    block_rows : number of lines per block
    """

    rdr_grid         = frame.getRadarGrid(freq)
    rg_pxl_spacing   = rdr_grid.range_pixel_spacing
    wavelength       = rdr_grid.wavelength
    rg_sample_freq   = isce3.core.speed_of_light / 2.0 / rg_pxl_spacing
    rg_bandwidth     = frame.getSwathMetadata(freq).processed_range_bandwidth
    center_frequency = isce3.core.speed_of_light / wavelength
    slant_range      = rdr_grid.slant_range
    
    diff_frequency = center_frequency - cfrequency
    height, width = slc.shape
    slc = np.array(slc,dtype='complex')
    nblocks = int(height / block_rows)
    if (nblocks == 0) :
        nblocks = 1
    elif (np.mod(height, (nblocks * block_rows)) != 0) :
        nblocks = nblocks + 1
    print("nblocks : ", nblocks)

    fft_size = width 
    window_target = constructRangeBandpassCosine( 
                    subBandCenterFrequencies = 0,
                    frequencyLH = [- rg_bandwidth/2, 
                                     rg_bandwidth/2],
                    samplingFrequency = rg_sample_freq, 
                    fft_size = fft_size,
                    beta = 0.25)

    bpfilter=constructRangeBandpassCosine(
                    subBandCenterFrequencies = center_frequency,
                    frequencyLH = [fL, fH],
                    samplingFrequency = rg_sample_freq, 
                    fft_size = fft_size,
                    beta = 0.25)
    
    resampling_factor = rg_bandwidth / int(np.abs(fH-fL))
    sub_fft_size = int(fft_size/resampling_factor)
    power_factor = resampling_factor /1.1
    print('resampling_factor : ',resampling_factor)

    filtered_slc = np.zeros([slc.shape[0], sub_fft_size],dtype=np.complex64)
    for block in range(0,nblocks):
        print("block: ",block)
        row_start = block * block_rows

        if ((row_start + block_rows) > height) :
            block_rows_data = height - row_start
        else :
            block_rows_data = block_rows
        
        slc_block = np.array(slc[row_start:row_start + block_rows_data,:])
        
        # remove the windowing effect from the spectrum 
        spectrum_target = fft(np.array(slc_block)) / np.tile(window_target,[block_rows_data,1])
        # apply new bandpass window to spectrum 
        slc_bp = ifft(spectrum_target * np.tile(bpfilter,[block_rows_data,1])*np.sqrt(power_factor))        
        # demodulate the SLC to be baseband to new center frequency
        slc_demodulate = demodulate_slc(slc_bp, diff_frequency, rg_sample_freq)

        # resample SLC 
        filtered_slc[row_start:row_start + block_rows_data,:] = \
                     resample(slc_demodulate, sub_fft_size ,axis=1)


    meta = dict()
    meta['centerFrequency'] = cfrequency
    meta['Bandwidth'] = fH - fL
    meta['rangeSpacing'] = rg_pxl_spacing * resampling_factor
    meta['slantRange'] =  np.linspace(slant_range(0),slant_range(fft_size),\
                          sub_fft_size,endpoint=False)

    return filtered_slc, meta

def demodulate_slc(SLC, diff_frequency, rg_sample_freq):
    """
    demodulate SLC for a given frequecy diff_frequency
    """
    height, width = np.shape(SLC)

    rangeTime=np.zeros(width)
    for ii in range(width):
        rangeTime[ii] = ii / rg_sample_freq
    
    slc_de = np.zeros([height,width],dtype=complex)
    for jj in range(height):
        slc_de[jj,:] = SLC[jj,:] * np.exp(-1 * 1j * 2.0 * np.pi * -1 * \
                        diff_frequency * rangeTime)
    return slc_de

def freqSpectrum(cfrequency, dt, fft_size):
    freq = cfrequency + fftfreq(fft_size, dt)
    return freq

def constructRangeBandpassCosine(subBandCenterFrequencies,
                                 samplingFrequency,
                                 fft_size,
                                 frequencyLH,
                                 beta):

    frequency = freqSpectrum(
                cfrequency = subBandCenterFrequencies,
                dt = 1.0 / samplingFrequency,
                fft_size = fft_size) 

    freq_size=np.shape(frequency)[0]
    norm = 1.0;    
    fL, fH = frequencyLH
    fmid = (fH + fL) * 0.5 
    subbandwidth = np.abs( fH - fL )
    df = 0.5 * subbandwidth * beta

    filter1D=np.zeros([freq_size],dtype='complex')
    for i in range(0,freq_size):

        #Get the absolute value of shifted frequency
        freq = np.abs(frequency[i])
        freqabs = np.abs(freq- fmid)

        # Passband
        if (freq <= (fH - df)) and (freq >= (fL + df)):
            filter1D[i] = 1

        # Transition region
        elif ((freq < (fL + df)) and (freq >= (fL - df))) \
         or ((freq <= (fH + df)) and (freq > (fH - df))):
            filter1D[i] = norm * 0.5 * (1.0 + np.cos(np.pi / (subbandwidth*beta) *
                          (freqabs - 0.5 * (1.0 - beta) * subbandwidth)))
    return filter1D
