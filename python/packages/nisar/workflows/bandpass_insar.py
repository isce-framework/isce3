#!/usr/bin/env python3

import os
import pathlib
import time

import h5py
import isce3
import journal
import numpy as np
from scipy.fft import next_fast_len

from isce3.io import HDF5OptimizedReader
from isce3.splitspectrum import splitspectrum
from nisar.h5 import cp_h5_meta_data
from nisar.products.insar.product_paths import CommonPaths
from nisar.products.readers import RSLC
from nisar.workflows.bandpass_insar_runconfig import BandpassRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse


def decimate_input_data_exception_mask(src_h5, dst_h5, freq_path,
                                       bandpassed_samples, decimation_factor,
                                       blocksize):
    '''
    Replace a frequency's inputDataExceptionMask with one on the
    bandpassed range grid.

    cp_h5_meta_data copies the mask verbatim, leaving it on the
    pre-bandpass grid while the SLC rasters and slantRange are rewritten
    on the decimated one, which puts the mask out of step with the swath
    it describes. This brings it back onto that grid.

    Input sample i belongs to output sample i // decimation_factor, the
    same mapping validSamplesSubSwath{i} is rescaled by, and the trailing
    input samples that form no whole group are dropped, as
    SplitSpectrum.bandpass_shift_spectrum drops them from the SLC. Every
    bit of each group is OR-reduced, so no bit set in any contributing
    input sample is lost. That is the aggregation the dataset is defined
    by, a "bitwise OR of input data exception codes ... also includes OR
    of validity mask bits", and it applies unchanged to the 8-bit mask,
    which carries anomaly codes alone, and to the 16-bit mask, which
    carries per-polarization validity bits in its high byte as well.

    The dtype, chunk shape (clamped to the narrower grid), compression
    and attributes of the source mask are preserved. A frequency with no
    mask is left alone, and so is a mask already on the bandpassed grid
    when decimation_factor is 1 and there is nothing to decimate.

    Parameters
    ----------
    src_h5 : h5py.File
        The opened HDF5 file of the RSLC being bandpassed
    dst_h5 : h5py.File
        The opened HDF5 file of the bandpassed RSLC, holding the verbatim
        copy of the mask that this function replaces
    freq_path : str
        The HDF5 path of the frequency group, e.g.
        '/science/LSAR/RSLC/swaths/frequencyA'
    bandpassed_samples : int
        Number of range samples of the bandpassed grid, i.e. the width
        the mask has to end up with. Read from the bandpassed SLC rather
        than recomputed here, so the mask cannot end up disagreeing with
        the rasters it describes
    decimation_factor : int
        Number of input samples per bandpassed sample, >= 1
    blocksize : int
        Number of lines per block read

    Raises
    ------
    ValueError
        If the mask is already as narrow as the bandpassed grid while a
        decimation is expected, or is too narrow to cover that grid.
        Either way the mask does not describe the source image pixel for
        pixel, so there is no sound way to put it on the bandpassed grid
    '''
    mask_path = f"{freq_path}/inputDataExceptionMask"
    if mask_path not in src_h5:
        return

    src_mask = src_h5[mask_path]
    lines, samples = src_mask.shape
    if samples == bandpassed_samples:
        if decimation_factor != 1:
            raise ValueError(
                f"{mask_path} is already {samples} samples wide, matching "
                f"the bandpassed grid, but a decimation factor of "
                f"{decimation_factor} was expected")
        else:
            return

    # Equals the resample_width_end that bandpass_shift_spectrum trims the
    # SLC to, so the mask drops the same trailing samples as the rasters
    samples_used = bandpassed_samples * decimation_factor
    if samples_used > samples:
        raise ValueError(
            f"{mask_path} has {samples} samples, too few to cover "
            f"{bandpassed_samples} bandpassed samples at a decimation "
            f"factor of {decimation_factor}")

    attrs = dict(src_mask.attrs)
    del dst_h5[mask_path]
    dst_mask = dst_h5.create_dataset(
        mask_path, (lines, bandpassed_samples), dtype=src_mask.dtype,
        chunks=tuple(min(c, n) for c, n in zip(src_mask.chunks or (128, 128),
                                               (lines, bandpassed_samples))),
        compression=src_mask.compression,
        compression_opts=src_mask.compression_opts)
    dst_mask.attrs.update(attrs)

    for row_start in range(0, lines, blocksize):
        row_stop = min(row_start + blocksize, lines)
        groups = src_mask[row_start:row_stop, :samples_used].reshape(
            row_stop - row_start, bandpassed_samples, decimation_factor)
        dst_mask.write_direct(np.bitwise_or.reduce(groups, axis=2),
                              dest_sel=np.s_[row_start:row_stop, :])


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
    ref_slc = RSLC(hdf5file=ref_hdf5)
    sec_slc = RSLC(hdf5file=sec_hdf5)

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
        base_meta_data = splitspectrum.BandpassMetaData.load_from_slc(
            slc_product=base_slc,
            freq=freq)
        target_meta_data = splitspectrum.BandpassMetaData.load_from_slc(
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
        with HDF5OptimizedReader(name=target_hdf5, mode='r',
                                 libver='latest', swmr=True) as src_h5, \
             HDF5OptimizedReader(name=target_output, mode='w') as dst_h5:

            # Copy HDF 5 file to be bandpassed
            cp_h5_meta_data(src_h5, dst_h5, f'{CommonPaths.RootPath}')

            for pol in pol_list:

                target_raster_str = \
                    f'HDF5:{target_hdf5}:/{target_slc.slcPath(freq, pol)}'
                target_slc_raster = isce3.io.Raster(target_raster_str)
                rows = target_slc_raster.length
                cols = target_slc_raster.width
                nblocks = int(np.ceil(rows / blocksize))
                if fft_size is None:
                    fft_size = next_fast_len(cols)

                reader = target_slc.getSlcDatasetAsNativeComplex(freq, pol)

                for block in range(0, nblocks):
                    print("-- bandpass block: ", block)
                    row_start = block * blocksize
                    if (row_start + blocksize > rows):
                        block_rows_data = rows - row_start
                    else:
                        block_rows_data = blocksize

                    dest_pol_path = f"{dest_freq_path}/{pol}"
                    # Read SLC from HDF5
                    target_slc_image = reader[
                        row_start:row_start + block_rows_data,
                        :]
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

            # Input samples per bandpassed sample, so input sample i maps to
            # output sample i // decimation_factor. The ratio has to be
            # rounded to the integer it must be, because the two spacings
            # are accumulated separately and land ~1e-11 apart: here
            # 3.1228381041437387 against 6.245676208333333, a ratio of
            # 0.499999999996329 rather than 0.5. Scaling indices by that
            # unrounded ratio loses a pixel on every even index, since the
            # exact i / 2 is a whole number that any downward error drops
            # below, and int() truncates:
            #   i=1602 -> 1602 * 0.499999999996329 = 800.9999999941 -> 800,
            #             one sample low of the correct 801
            #   i=1601 -> 1601 * 0.499999999996329 = 800.4999999941 -> 800,
            #             correct, as odd indices land on x.5 and survive
            decimation_factor = int(np.round(
                bandpass_meta['range_spacing'] /
                target_meta_data.rg_pxl_spacing))

            # Handle the case when the decimation_factor == 0
            decimation_factor = max(decimation_factor, 1)

            subswath_number = src_h5[f"{dest_freq_path}/numberOfSubSwaths"][()]
            for swath_count in range(subswath_number):
                # Update the validateSamplesSubswaths
                valid_sample_path = \
                f"{dest_freq_path}/validSamplesSubSwath{swath_count + 1}"
                valid_samples = src_h5[valid_sample_path][()]
                data = dst_h5[f"{valid_sample_path}"]
                data[...] = valid_samples // decimation_factor

            # Width taken from a raster already written here, which the
            # slantRange below is sized from too, so all three agree
            decimate_input_data_exception_mask(
                src_h5, dst_h5, dest_freq_path,
                dst_h5[dest_pol_path].shape[1],
                decimation_factor, blocksize)

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
