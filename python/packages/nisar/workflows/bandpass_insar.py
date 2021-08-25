#!/usr/bin/env python3

import pathlib
import time
import os

import h5py
import journal
import numpy as np

import isce3
from nisar.products.readers import SLC
from nisar.workflows import h5_prep
from nisar.workflows.bandpass_insar_runconfig import BandpassRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.h5 import cp_h5_meta_data
import nisar.workflows.splitspectrum as splitspectrum

def run(cfg: dict):
    '''
    run bandpass
    '''
    # pull parameters from cfg
    ref_hdf5 = cfg['InputFileGroup']['InputFilePath']
    sec_hdf5 = cfg['InputFileGroup']['SecondaryFilePath']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    blocksize = cfg['processing']['blocksize']['y']

    scratch_path = pathlib.Path(cfg['ProductPathGroup']['ScratchPath'])

    # init parameters shared by frequency A and B
    ref_slc = SLC(hdf5file=ref_hdf5)
    sec_slc = SLC(hdf5file=sec_hdf5)

    error_channel = journal.error('bandpass.run')
    info_channel = journal.info("bandpass.run")
    info_channel.log("starting bandpass")

    t_all = time.time()

    # check if bandpass is necessary
    bandpass_modes = splitspectrum.insar_mixmode(ref_slc, sec_slc, freq_pols)

    # check if user provided path to raster(s) is a file or directory
    bandpass_slc_path =  pathlib.Path(f"{scratch_path}/bandpass/")

    if bandpass_modes:
        ref_slc_output = f"{scratch_path}/bandpass/ref_bp.h5"
        sec_slc_output = f"{scratch_path}/bandpass/sec_bp.h5"   
        bandpass_slc_path.mkdir(parents=True, exist_ok=True)

    common_parent_path = 'science/LSAR'
    
    # freq: [A, B], target : 'ref' or 'sec'
    for freq, target in bandpass_modes.items():

        bandpass = splitspectrum.Splitspectrum()
        pol_list = freq_pols[freq]

        # if reference has wide bandwidth, then reference will be bandpassed
        # base   : SLC to be referenced  
        # target : SLC to be bandpassed
        if target == 'ref':

            base_hdf5   = sec_hdf5
            target_hdf5 = ref_hdf5
            target_slc  = ref_slc
            base_slc    = sec_slc

            # update reference SLC path 
            cfg['InputFileGroup']['InputFilePath'] = ref_slc_output
            target_output = ref_slc_output

        elif target == 'sec':

            base_hdf5   = ref_hdf5
            target_hdf5 = sec_hdf5
            target_slc  = sec_slc
            base_slc    = ref_slc

            # update secondary SLC path 
            cfg['InputFileGroup']['SecondaryFilePath'] = sec_slc_output
            target_output = sec_slc_output

        if os.path.exists(target_output):
            os.remove(target_output)

        # Copy HDF 5 file to be bandpassed    
        with h5py.File(target_hdf5, 'r', libver='latest', swmr=True) as src_h5, \
             h5py.File(target_output, 'w', libver='latest', swmr=True) as dst_h5:
            cp_h5_meta_data(src_h5, dst_h5, f'{common_parent_path}')

        ## meta data extraction 
        base_meta_data = splitspectrum.meta_data_bandpass(base_slc, freq)
        bandwidth_half = base_meta_data['rg_bandwidth'] * 0.5
        low_frequency_base = base_meta_data['center_frequency'] - bandwidth_half
        high_frequency_base = base_meta_data['center_frequency'] + bandwidth_half

        dest_freq_path=f"/science/LSAR/SLC/swaths/frequency{freq}"
        with h5py.File(target_hdf5, 'r', libver='latest', swmr=True) as src_h5, \
            h5py.File(target_output, 'r+') as dst_h5:
            for pol in pol_list:
                
                target_raster_str = f'HDF5:{target_hdf5}:/{target_slc.slcPath(freq, pol)}'
                target_slc_raster = isce3.io.Raster(target_raster_str)   
                cols = target_slc_raster.length         
                nblocks  = int( cols / blocksize)

                bandpass.frame = target_slc
                bandpass.low_frequency = low_frequency_base
                bandpass.high_frequency = high_frequency_base
                bandpass.new_center_frequency = base_meta_data['center_frequency']
                bandpass.freq = freq

                if (nblocks == 0) :
                    nblocks = 1
                elif (np.mod(cols, (nblocks * blocksize)) != 0) :
                    nblocks = nblocks + 1
  
                for block in range(0,nblocks):
                    print("-- bandpass block: ",block)
                    row_start = block * blocksize
                    if ((row_start + blocksize) > cols) :
                        block_rows_data = cols - row_start
                    else :
                        block_rows_data = blocksize
                    
                    dest_pol_path=f"{dest_freq_path}/{pol}"
                    target_slc_image = src_h5[dest_pol_path][row_start:row_start + block_rows_data,:]
                    bandpass.slc_raster = target_slc_image
                    bandpass_slc, meta = bandpass.bandpass_spectrum()
      
                    if block == 0:
                        del dst_h5[dest_pol_path]
                        dst_h5.create_dataset(dest_pol_path, 
                                        np.shape(bandpass_slc), 
                                        np.complex64,
                                        chunks=(128, 128),
                                        data= np.complex64(bandpass_slc),
                                        maxshape=(None,None))
                    else:
                        dst_h5[dest_pol_path].resize((dst_h5[dest_pol_path].shape[0] + bandpass_slc.shape[0]), axis = 0)
                        dst_h5[dest_pol_path][-bandpass_slc.shape[0]:] = np.complex64(bandpass_slc)

                dst_h5[dest_pol_path].attrs['description'] = f"Bandpass SLC image ({pol})"
                dst_h5[dest_pol_path].attrs['units'] = f"DN"
 
        # update meta information for bandpass SLC
        with h5py.File(target_output, 'r+', libver='latest', swmr=True) as dst_h5:

            data = dst_h5[f"{dest_freq_path}/processedCenterFrequency"]
            data[...] = meta['centerFrequency']
            data = dst_h5[f"{dest_freq_path}/slantRangeSpacing"]
            data[...] = meta['rangeSpacing']
            data = dst_h5[f"{dest_freq_path}/processedRangeBandwidth"]
            data[...] = base_meta_data['rg_bandwidth'] 
            del dst_h5[f"{dest_freq_path}/slantRange"]
            dst_h5.create_dataset(f"{dest_freq_path}/slantRange",data=meta['slantRange'])

    # update HDF5 
    if bandpass_modes:
        h5_prep.run(cfg)

    t_all_elapsed = time.time() - t_all
    print ('total processing time: ', t_all_elapsed, ' sec')
    info_channel.log(
        f"successfully ran bandpass in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    '''
    run bandpass from command line
    '''
    # load command line args
    bandpass_parser = YamlArgparse()
    args = bandpass_parser.parse()
    # get a runconfig dict from command line args
    badnpass_runconfig = BandpassRunConfig(args)
    # run bandpass
    run(bandpass_runconfig.cfg)
