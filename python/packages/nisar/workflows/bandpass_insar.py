#!/usr/bin/env python3

import os
import pathlib
import time

import h5py
import isce3
import journal
import numpy as np
from isce3.splitspectrum import splitspectrum
from nisar.h5 import cp_h5_meta_data
from nisar.products.insar.product_paths import CommonPaths
from nisar.products.readers import SLC
from nisar.workflows.bandpass_insar_runconfig import BandpassRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg: dict):
    '''
    run bandpass
    '''
    # pull parameters from cfg
    ref_hdf5 = cfg['input_file_group']['reference_rslc_file']
    sec_hdf5 = cfg['input_file_group']['secondary_rslc_file']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    blocksize = cfg['processing']['bandpass']['lines_per_block']
    window_function = cfg['processing']['bandpass']['window_function']
    window_shape = cfg['processing']['bandpass']['window_shape']
    fft_size = cfg['processing']['bandpass']['range_fft_size']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])

    # init parameters shared by frequency A and B
    ref_slc = SLC(hdf5file=ref_hdf5)
    sec_slc = SLC(hdf5file=sec_hdf5)

    info_channel = journal.info("bandpass_insar.run")
    info_channel.log("starting bandpass_insar")

    t_all = time.time()

    # check if bandpass is necessary
    bandpass_modes = splitspectrum.check_range_bandwidth_overlap(
        ref_slc=ref_slc,
        sec_slc=sec_slc,
        pols=freq_pols)

    # check if user provided path to raster(s) is a file or directory
    bandpass_slc_path = pathlib.Path(f"{scratch_path}/bandpass/")

    if bandpass_modes:
        ref_slc_output = f"{bandpass_slc_path}/ref_slc_bandpassed.h5"
        sec_slc_output = f"{bandpass_slc_path}/sec_slc_bandpassed.h5"
        bandpass_slc_path.mkdir(parents=True, exist_ok=True)

    # freq: [A, B], target : 'ref' or 'sec'
    for freq, target in bandpass_modes.items():
        pol_list = freq_pols[freq]

        # if reference has a wider bandwidth, then reference will be bandpassed
        # base : SLC to be referenced
        # target : SLC to be bandpassed
        if target == 'ref':
            target_hdf5 = ref_hdf5
            target_slc = ref_slc
            base_slc = sec_slc

            # update reference SLC path
            cfg['input_file_group']['reference_rslc_file'] = ref_slc_output
            target_output = ref_slc_output

        elif target == 'sec':
            target_hdf5 = sec_hdf5
            target_slc = sec_slc
            base_slc = ref_slc

            # update secondary SLC path
            cfg['input_file_group']['secondary_rslc_file'] = sec_slc_output
            target_output = sec_slc_output

        if os.path.exists(target_output):
            os.remove(target_output)

        # meta data extraction
        base_meta_data = splitspectrum.bandpass_meta_data.load_from_slc(
            slc_product=base_slc,
            freq=freq)
        target_meta_data = splitspectrum.bandpass_meta_data.load_from_slc(
            slc_product=target_slc,
            freq=freq)

        sampling_bandwidth_ratio = \
            base_meta_data.rg_sample_freq / base_meta_data.rg_bandwidth

        info_channel.log("base RSLC:")
        info_channel.log(f"    bandwidth : {base_meta_data.rg_bandwidth}")
        info_channel.log(f"    sampling_frequency : {base_meta_data.rg_sample_freq}")
        info_channel.log("target RSLC:")
        info_channel.log(f"    bandwidth : {target_meta_data.rg_bandwidth}")
        info_channel.log(f"    sampling_frequency : {target_meta_data.rg_sample_freq}")
        info_channel.log(f"sampling_frequency / bandwidth : {sampling_bandwidth_ratio}")

        bandwidth_half = 0.5 * base_meta_data.rg_bandwidth
        low_frequency_base = \
            base_meta_data.center_freq - bandwidth_half
        high_frequency_base = \
            base_meta_data.center_freq + bandwidth_half

        # Initialize bandpass instance
        # Specify meta parameters of SLC to be bandpassed
        bandpass = splitspectrum.SplitSpectrum(
            rg_sample_freq=target_meta_data.rg_sample_freq,
            rg_bandwidth=target_meta_data.rg_bandwidth,
            center_frequency=target_meta_data.center_freq,
            slant_range=target_meta_data.slant_range,
            freq=freq,
            sampling_bandwidth_ratio=sampling_bandwidth_ratio)
        swath_path = ref_slc.SwathPath
        dest_freq_path = f"{swath_path}/frequency{freq}"
        with h5py.File(target_hdf5, 'r', libver='latest',
                       swmr=True) as src_h5, \
            h5py.File(target_output, 'w') as dst_h5:
            # Copy HDF 5 file to be bandpassed
            cp_h5_meta_data(src_h5, dst_h5, f'{CommonPaths.RootPath}')

            for pol in pol_list:

                target_raster_str = \
                    f'HDF5:{target_hdf5}:/{target_slc.slcPath(freq, pol)}'
                target_slc_raster = isce3.io.Raster(target_raster_str)
                rows = target_slc_raster.length
                cols = target_slc_raster.width
                nblocks = int(np.ceil(rows / blocksize))

                for block in range(0, nblocks):
                    print("-- bandpass block: ", block)
                    row_start = block * blocksize
                    if (row_start + blocksize > rows):
                        block_rows_data = rows - row_start
                    else:
                        block_rows_data = blocksize

                    dest_pol_path = f"{dest_freq_path}/{pol}"
                    target_slc_image = np.empty([block_rows_data, cols],
                                                dtype=complex)
                    # Read SLC from HDF5
                    src_h5[dest_pol_path].read_direct(
                        target_slc_image,
                        np.s_[row_start:row_start + block_rows_data, :])

                    # Specify low and high frequency to be passed (bandpass)
                    # and the center frequency to be basebanded (demodulation)
                    bandpass_slc, bandpass_meta = \
                        bandpass.bandpass_shift_spectrum(
                            slc_raster=target_slc_image,
                            low_frequency=low_frequency_base,
                            high_frequency=high_frequency_base,
                            new_center_frequency=base_meta_data.center_freq,
                            fft_size=fft_size,
                            window_shape=window_shape,
                            window_function=window_function,
                            resampling=True
                            )

                    if block == 0:
                        del dst_h5[dest_pol_path]
                        # Initialize the raster with updated shape in HDF5
                        dst_h5.create_dataset(dest_pol_path,
                                              [rows, np.shape(bandpass_slc)[1]],
                                              np.complex64, chunks=(128, 128))
                    # Write bandpassed SLC to HDF5
                    dst_h5[dest_pol_path].write_direct(
                        bandpass_slc,
                        dest_sel=np.s_[row_start:row_start + block_rows_data,
                                       :])

                dst_h5[dest_pol_path].attrs['description'] = \
                    f"Bandpass SLC image ({pol})"
                dst_h5[dest_pol_path].attrs['units'] = ""

            bandpass_ratio = \
                target_meta_data.rg_pxl_spacing / bandpass_meta['range_spacing']
            subswath_number = src_h5[f"{dest_freq_path}/numberOfSubSwaths"][()]
            for swath_count in range(subswath_number):
                # Update the validateSamplesSubswaths
                valid_sample_path = \
                f"{dest_freq_path}/validSamplesSubSwath{swath_count + 1}"
                valid_samples = src_h5[valid_sample_path][()]
                valid_samples_bandpass = \
                    np.array(valid_samples * bandpass_ratio, dtype='int')
                data = dst_h5[f"{valid_sample_path}"]
                data[...] = valid_samples_bandpass.astype(int)

            # update meta information for bandpass SLC
            data = dst_h5[f"{dest_freq_path}/processedCenterFrequency"]
            data[...] = bandpass_meta['center_frequency']
            data = dst_h5[f"{dest_freq_path}/slantRangeSpacing"]
            data[...] = bandpass_meta['range_spacing']
            data = dst_h5[f"{dest_freq_path}/processedRangeBandwidth"]
            data[...] = base_meta_data.rg_bandwidth
            del dst_h5[f"{dest_freq_path}/slantRange"]
            dst_h5.create_dataset(f"{dest_freq_path}/slantRange",
                                  data=bandpass_meta['slant_range'])

    t_all_elapsed = time.time() - t_all
    print('total processing time: ', t_all_elapsed, ' sec')
    info_channel.log(
        f"successfully ran bandpass_insar in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    '''
    run bandpass from command line
    '''
    # load command line args
    bandpass_parser = YamlArgparse()
    args = bandpass_parser.parse()
    # get a runconfig dict from command line args
    bandpass_runconfig = BandpassRunConfig(args)
    # run bandpass
    run(bandpass_runconfig.cfg)
