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
from nisar.products.readers import SLC

from nisar.workflows.split_spectrum_runconfig import SplitSpectrumRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse


def prep_subband_h5(full_hdf5: str,
                    sub_band_hdf5: str,
                    freq_pols):

    """Prepare HDF5 file for sub-band SLCs by copying data from
     source SLC HDF5 file.

    Parameters
    ----------
    full_hdf5 : str
        path of source SLC HDF5 file
    sub_band_hdf5 : str
        path of sub-band SLC HDF5 file
    freq_pols : dict
        list of polarzations for frequency A and B

    """
    full_slc = SLC(hdf5file=full_hdf5)
    common_parent_path = 'science/LSAR'
    swath_path = full_slc.SwathPath
    metadata_path = full_slc.MetadataPath
    ident_path = f'{common_parent_path}/identification'

    with h5py.File(full_hdf5, 'r', libver='latest', swmr=True) as src_h5, \
        h5py.File(sub_band_hdf5, 'w') as dst_h5:

        for freq_ab in ['A', 'B']:
            freq_key = f'frequency{freq_ab}'

            if freq_key not in src_h5[swath_path]:
                continue

            freq_path = f'{swath_path}/{freq_key}'
            pol_path = f'{freq_path}/listOfPolarizations'
            pol_list = np.array(src_h5[pol_path][()], dtype=str) \
                    if freq_key in src_h5[swath_path] else [""]

            if len(pol_list) and (freq_pols[freq_ab] is not None):
                pols_excludes = [pol for pol in pol_list
                    if pol not in freq_pols[freq_ab]]
            else:
                pols_excludes = list(pol_list)

            if len(pols_excludes):
                cp_h5_meta_data(src_h5, dst_h5, freq_path,
                        excludes=pols_excludes)
            else:
                cp_h5_meta_data(src_h5, dst_h5, freq_path,
                        excludes=[''])

        cp_h5_meta_data(src_h5, dst_h5, metadata_path,
                    excludes=[''])
        cp_h5_meta_data(src_h5, dst_h5, ident_path,
                    excludes=[''])
        cp_h5_meta_data(src_h5, dst_h5, swath_path,
                    excludes=['frequencyA', 'frequencyB'])

def run(cfg: dict):
    '''
    run split spectrum
    '''
    # pull parameters from cfg
    ref_hdf5 = cfg['input_file_group']['reference_rslc_file_path']
    sec_hdf5 = cfg['input_file_group']['secondary_rslc_file_path']

    # Extract range split spectrum dictionary and corresponding parameters
    ionosphere_option = cfg['processing']['ionosphere_phase_correction']
    method = ionosphere_option['spectral_diversity']
    split_cfg = ionosphere_option['split_range_spectrum']
    iono_freq_pol = ionosphere_option['list_of_frequencies']
    blocksize = split_cfg['lines_per_block']
    window_function = split_cfg['window_function']
    window_shape = split_cfg['window_shape']
    low_band_bandwidth = split_cfg['low_band_bandwidth']
    high_band_bandwidth = split_cfg['high_band_bandwidth']

    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])

    info_channel = journal.info("split_spectrum.run")
    info_channel.log("starting split_spectrum")

    t_all = time.time()

    # Check split spectrum method
    if method == 'split_main_band':
        split_band_path = pathlib.Path(
            f"{scratch_path}/ionosphere/split_spectrum/")
        split_band_path.mkdir(parents=True, exist_ok=True)

        common_parent_path = 'science/LSAR'
        freq = 'A'
        pol_list = iono_freq_pol[freq]
        info_channel.log(f'Split the main band {pol_list} of the signal')

        for hdf5_ind, hdf5_str in enumerate([ref_hdf5, sec_hdf5]):
            # reference SLC
            if hdf5_ind == 0:
                low_band_output = f"{split_band_path}/ref_low_band_slc.h5"
                high_band_output = f"{split_band_path}/ref_high_band_slc.h5"
            # secondary SLC
            else:
                low_band_output = f"{split_band_path}/sec_low_band_slc.h5"
                high_band_output = f"{split_band_path}/sec_high_band_slc.h5"
            # Open RSLC product
            slc_product = SLC(hdf5file=hdf5_str)
            # meta data extraction
            meta_data = splitspectrum.bandpass_meta_data.load_from_slc(
                slc_product=slc_product,
                freq=freq)
            bandwidth_half = 0.5 * meta_data.rg_bandwidth
            low_frequency_slc = meta_data.center_freq - bandwidth_half
            high_frequency_slc = meta_data.center_freq + bandwidth_half

            # first and second elements are the frequency ranges for low and high sub-bands, respectively.
            low_subband_frequencies = np.array(
                [low_frequency_slc, low_frequency_slc + low_band_bandwidth])
            high_subband_frequencies = np.array(
                [high_frequency_slc - high_band_bandwidth, high_frequency_slc])

            low_band_center_freq = low_frequency_slc + low_band_bandwidth/2
            high_band_center_freq = high_frequency_slc - high_band_bandwidth/2
            # Specify split-spectrum parameters
            split_spectrum_parameters = splitspectrum.SplitSpectrum(
                rg_sample_freq=meta_data.rg_sample_freq,
                rg_bandwidth=meta_data.rg_bandwidth,
                center_frequency=meta_data.center_freq,
                slant_range=meta_data.slant_range,
                freq=freq)

            dest_freq_path = os.path.join(slc_product.SwathPath,
                                          f'frequency{freq}')

            # prepare HDF5 for subband SLC HDF5
            prep_subband_h5(hdf5_str, low_band_output, iono_freq_pol)
            prep_subband_h5(hdf5_str, high_band_output, iono_freq_pol)

            with h5py.File(hdf5_str, 'r', libver='latest', swmr=True) as src_h5, \
                    h5py.File(low_band_output, 'r+') as dst_h5_low, \
                    h5py.File(high_band_output, 'r+') as dst_h5_high:

                # Copy HDF5 metadata for low high band
                for pol in pol_list:
                    raster_str = f'HDF5:{hdf5_str}:/{slc_product.slcPath(freq, pol)}'
                    slc_raster = isce3.io.Raster(raster_str)
                    rows = slc_raster.length
                    cols = slc_raster.width
                    nblocks = int(np.ceil(rows / blocksize))
                    fft_size = cols

                    for block in range(0, nblocks):
                        info_channel.log(f" split_spectrum block: {block}")
                        row_start = block * blocksize
                        if ((row_start + blocksize) > rows):
                            block_rows_data = rows - row_start
                        else:
                            block_rows_data = blocksize

                        dest_pol_path = f"{dest_freq_path}/{pol}"
                        target_slc_image = np.empty([block_rows_data, cols],
                                                    dtype=complex)

                        src_h5[dest_pol_path].read_direct(
                            target_slc_image,
                            np.s_[row_start: row_start + block_rows_data, :])

                        subband_slc_low, subband_meta_low = \
                            split_spectrum_parameters.bandpass_shift_spectrum(
                            slc_raster=target_slc_image,
                            low_frequency=low_subband_frequencies[0],
                            high_frequency=low_subband_frequencies[1],
                            new_center_frequency=low_band_center_freq,
                            fft_size=fft_size,
                            window_shape=window_shape,
                            window_function=window_function,
                            resampling=False
                        )

                        subband_slc_high, subband_meta_high = \
                            split_spectrum_parameters.bandpass_shift_spectrum(
                            slc_raster=target_slc_image,
                            low_frequency=high_subband_frequencies[0],
                            high_frequency=high_subband_frequencies[1],
                            new_center_frequency=high_band_center_freq,
                            fft_size=fft_size,
                            window_shape=window_shape,
                            window_function=window_function,
                            resampling=False
                        )
                        if block == 0:
                            del dst_h5_low[dest_pol_path]
                            del dst_h5_high[dest_pol_path]
                            # Initialize the raster with updated shape in HDF5
                            dst_h5_low.create_dataset(dest_pol_path,
                                                      [rows, cols],
                                                      np.complex64,
                                                      chunks=(128, 128))
                            dst_h5_high.create_dataset(dest_pol_path,
                                                       [rows, cols],
                                                       np.complex64,
                                                       chunks=(128, 128))

                        # Write bandpassed SLC to HDF5
                        dst_h5_low[dest_pol_path].write_direct(
                            subband_slc_low,
                            dest_sel=np.s_[
                                row_start: row_start + block_rows_data, :])

                        dst_h5_high[dest_pol_path].write_direct(
                            subband_slc_high,
                            dest_sel=np.s_[
                                row_start: row_start + block_rows_data, :])

                    dst_h5_low[dest_pol_path].attrs[
                        'description'] = f"Split-spectrum SLC image ({pol})"
                    dst_h5_low[dest_pol_path].attrs['units'] = f""

                    dst_h5_high[dest_pol_path].attrs[
                        'description'] = f"Split-spectrum SLC image ({pol})"
                    dst_h5_high[dest_pol_path].attrs['units'] = f""

                # update meta information for bandpass SLC
                data = dst_h5_low[f"{dest_freq_path}/processedCenterFrequency"]
                data[...] = subband_meta_low['center_frequency']
                data = dst_h5_low[f"{dest_freq_path}/processedRangeBandwidth"]
                data[...] = subband_meta_low['rg_bandwidth']
                data = dst_h5_high[f"{dest_freq_path}/processedCenterFrequency"]
                data[...] = subband_meta_high['center_frequency']
                data = dst_h5_high[f"{dest_freq_path}/processedRangeBandwidth"]
                data[...] = subband_meta_high['rg_bandwidth']
    else:
        info_channel.log('Split spectrum is not needed')

    t_all_elapsed = time.time() - t_all
    info_channel.log(
        f"successfully ran split_spectrum in {t_all_elapsed:.3f} seconds")

if __name__ == "__main__":
    '''
    Run split-spectrum from command line
    '''
    # load command line args
    split_spectrum_parser = YamlArgparse()
    args = split_spectrum_parser.parse()

    # get a runconfig dict from command line args
    split_spectrum_runconfig = SplitSpectrumRunConfig(args)

    # run bandpass
    run(split_spectrum_runconfig.cfg)
