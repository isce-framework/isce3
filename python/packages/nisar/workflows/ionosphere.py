#!/usr/bin/env python3
import copy
import os
import pathlib
import shutil
import time
from itertools import repeat

import isce3
import journal
import numpy as np
from isce3.atmosphere.ionosphere_filter import (
    IonosphereFilter, read_block_array, unwrapping_correction_with_filter,
    write_array)
from isce3.atmosphere.main_band_estimation import (
    MainDiffMsBandIonosphereEstimation, MainSideBandIonosphereEstimation)
from isce3.atmosphere.split_band_estimation import (
    LowHighSubbandIonosphereEstimation,
    MainDiffLowHighSubbandIonosphereEstimation)
from isce3.core.block_param_generator import (block_param_generator,
                                              get_raster_block,
                                              write_raster_block)
from isce3.io import HDF5OptimizedReader
from isce3.signal.interpolate_by_range import (decimate_freq_a_array,
                                               interpolate_freq_b_array)
from isce3.splitspectrum import splitspectrum
from isce3.unwrap.bridge_phase import bridge_unwrapped_phase
from isce3.unwrap.preprocess import project_map_to_radar
from nisar.products.insar.product_paths import (CommonPaths, RIFGGroupsPaths,
                                                RUNWGroupsPaths)
from nisar.products.readers import SLC
from nisar.products.utils import deepcopy_runconfig_and_keep_isce3_obj
from nisar.workflows import (crossmul, filter_interferogram, h5_prep,
                             prepare_insar_hdf5, resample_slc_v2, unwrap)
from nisar.workflows.compute_stats import compute_stats_real_hdf5_dataset
from nisar.workflows.ionosphere_runconfig import InsarIonosphereRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from osgeo import gdal


def write_disp_block_hdf5(
        hdf5_path,
        path,
        data,
        rows,
        block_row=0):
    """write block array to HDF5
    Parameters
    ----------
    hdf5_path : str
        output HDF5 file name
    path : str
        HDF5 path for dataset
    data : numpy.ndarray
        block data to be saved to HDF5
    rows : int
        number of rows of entire data
    block_row : int
        block start index
    """
    error_channel = journal.error('ionosphere.write_disp_block_hdf5')
    if not os.path.isfile(hdf5_path):
        err_str = f"{hdf5_path} not found"
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)
    with HDF5OptimizedReader(name=hdf5_path, mode='r+') as dst_h5:
        block_length, block_width = data.shape
        dst_h5[path].write_direct(
            data,
            dest_sel=np.s_[block_row:block_row + block_length,
                           :block_width])


def compute_phase_jump(previous_with_pad, current_image, half_pad_length):
    """
    Compute and correct for phase jumps between overlapping regions of two
    images.

    This function extracts the overlapping regions between a padded previous
    image and a current image, computes the phase jump (difference in phase),
    and applies the correction to the current image to account for phase
    unwrapping errors.

    Parameters
    ----------
    previous_with_pad : numpy.ndarray
        A 2D array representing the previous image, padded to account for the
        overlapping region. The overlap is assumed to be at the bottom of this
        image.
    current_image : numpy.ndarray
        A 2D array representing the current image. The overlap is assumed to
        be at the top of this image.
    half_pad_length : int
        The number of rows corresponding to the overlap region between the
        previous and current images.

    Returns
    -------
    current_image : numpy.ndarray
        The current image with the phase jump corrected.
    difference_jump : float
        The computed phase jump value, which was applied to correct the
        current image.
    """
    # Extract the overlap regions
    previous_overlap = previous_with_pad[-half_pad_length:, :]
    current_overlap = current_image[:half_pad_length, :]

    # Calculate the difference in phase in the overlap areas
    difference = np.nanmedian(current_overlap) - np.nanmedian(previous_overlap)

    # Calculate the jump in phase
    difference_jump = -((np.abs(difference) + np.pi) // (2.0 * np.pi))

    # Apply the correction to the current image
    current_image += 2.0 * np.pi * difference_jump

    return current_image, difference_jump


def decimate_freq_a_offset(iono_insar_cfg, original_dict):
    """Decimate range and azimuth offsets

    Parameters
    ----------
    iono_insar_cfg : dict
        dictionary of runconfigs
    original_dict: dict
        dictionary containing following parameters
        - scratch_path
        - reference_rslc_file
        - secondary_rslc_file
        - coregistered_slc_path
        - list_of_frequencies
        - output_runw
        - offsets_dir
    """
    # parameters
    blocksize = iono_insar_cfg['processing']['ionosphere_phase_correction'][
                'lines_per_block']

    offsets_dir = original_dict['offsets_dir']

    ref_slc_path = original_dict['reference_rslc_file']
    if iono_insar_cfg['processing']['fine_resample']['enabled']:
        resample_type = 'fine'
    else:
        resample_type = 'coarse'
    decimated_offset_dir = offsets_dir

    # Instantiate a RSLC Swath object to get slant range for frequency A and B
    slc_swath_obj_freqa = isce3.product.Swath(ref_slc_path, 'A')
    slc_swath_obj_freqb = isce3.product.Swath(ref_slc_path, 'B')

    main_slant = np.array(slc_swath_obj_freqa.slant_range)
    spacing_main = slc_swath_obj_freqa.range_pixel_spacing
    side_slant = np.array(slc_swath_obj_freqb.slant_range)
    spacing_side = slc_swath_obj_freqb.range_pixel_spacing

    resampling_scale_factor = float(int(np.round(spacing_side / spacing_main)))
    if resample_type == 'coarse':
        decimate_list = ['coarse']
    elif resample_type == 'fine':
        decimate_list = ['coarse', 'fine']

    for decimate_proc in decimate_list:
        if decimate_proc == 'coarse':
            coarse_offset_path = '/geo2rdr/freqA'
            coarse_offset_b_path = '/geo2rdr/freqB'

            offsets_path = f'{offsets_dir}/{coarse_offset_path}'
            offsets_b_path = f'{decimated_offset_dir}/{coarse_offset_b_path}'
        else:
            # We checked the existence of HH/VV offsets in resample_slc_runconfig.py
            # Select the first offsets available between HH and VV
            fine_offset_path = 'rubbersheet_offsets/freqA'
            fine_offset_b_path = 'rubbersheet_offsets/freqB'

            freq_offsets_path = f'{offsets_dir}/{fine_offset_path}'
            freq_offsets_b_path = f'{decimated_offset_dir}/{fine_offset_b_path}'

            if os.path.isdir(str(f'{freq_offsets_path}/HH')):
                offsets_path = f'{freq_offsets_path}/HH'
                offsets_b_path = f'{freq_offsets_b_path}/HH'
            else:
                offsets_path = f'{freq_offsets_path}/VV'
                offsets_b_path = f'{freq_offsets_b_path}/VV'

        rg_off_path = str(f'{offsets_path}/range.off')
        az_off_path = str(f'{offsets_path}/azimuth.off')

        rg_b_off_path = str(f'{offsets_b_path}/range.off')
        az_b_off_path = str(f'{offsets_b_path}/azimuth.off')

        # create new offset directory in ionosphere scratch
        os.makedirs(offsets_b_path, exist_ok=True)

        # open raster as GDAL datasets for decimation
        rg_off_obj = gdal.Open(rg_off_path)
        az_off_obj = gdal.Open(az_off_path)

        band = rg_off_obj.GetRasterBand(1)
        datatype = band.DataType
        # get dimensions for block processing
        rows_main = rg_off_obj.RasterYSize
        cols_main = rg_off_obj.RasterXSize
        nblocks = int(np.ceil(rows_main / blocksize))

        for off_obj, b_off_path in zip([rg_off_obj, az_off_obj],
                                       [rg_b_off_path, az_b_off_path]):
            off_scale_factor = [resampling_scale_factor
                                if 'range' in b_off_path else 1]

            for block in range(0, nblocks):
                row_start = block * blocksize
                if (row_start + blocksize > rows_main):
                    block_rows_data = rows_main - row_start
                else:
                    block_rows_data = blocksize

                offset_arr = off_obj.ReadAsArray(0, row_start,
                                                 cols_main,
                                                 block_rows_data)

                off_side = decimate_freq_a_array(
                                main_slant,
                                side_slant,
                                offset_arr) / off_scale_factor

                rows_output, cols_output = off_side.shape
                write_array(b_off_path,
                            off_side,
                            data_type=datatype,
                            block_row=row_start,
                            data_shape=[rows_main, cols_output],
                            file_type='ENVI')


def copy_iono_datasets(iono_insar_cfg,
                       input_runw,
                       output_runw,
                       blocksize,
                       oversample_flag=False,
                       slant_main=None,
                       slant_side=None):
    """copy ionosphere layers (frequency B) to frequency A of RUNW product
    with oversampling

    Parameters
    ----------
    iono_insar_cfg : dict
        dictionary of runconfigs
    input_runw : str
        file path of frequency B RUNW
    output_runw :str
        file path of frequency A RUNW
    oversample_flag: bool
        bool option for oversample
    slant_main : numpy.ndarray
        slant range array of frequency A band
    slant_side : numpy.ndarray
        slant range array of frequency B band
    oversample_flag : bool
    """

    iono_args = iono_insar_cfg['processing']['ionosphere_phase_correction']
    iono_freq_pols = iono_args['list_of_frequencies']

    # Instantiate RUNW object to easily access RUNW datasets
    runw_obj = RUNWGroupsPaths()
    swath_path = runw_obj.SwathsPath

    if oversample_flag:
        freq = 'A'
    else:
        freq = 'B'

    with HDF5OptimizedReader(name=input_runw, mode='a',
                             libver='latest', swmr=True) as src_h5, \
         HDF5OptimizedReader(name=output_runw, mode='a',
                             libver='latest', swmr=True) as dst_h5:

        pol_list = iono_freq_pols['A']
        for pol in pol_list:
            src_freq_path = f"{swath_path}/frequencyB"
            src_pol_path = f"{src_freq_path}/interferogram/{pol}"
            src_iono_path = f'{src_pol_path}/ionospherePhaseScreen'
            src_iono_unct_path = f'{src_pol_path}/ionospherePhaseScreenUncertainty'

            dest_freq_path = f"{swath_path}/frequency{freq}"
            dest_pol_path = f"{dest_freq_path}/interferogram/{pol}"
            iono_path = f'{dest_pol_path}/ionospherePhaseScreen'
            iono_unct_path = f'{dest_pol_path}/ionospherePhaseScreenUncertainty'

            freq_path = f'{swath_path}/frequency{freq}'
            ifg_path = f'{swath_path}/frequency{freq}/interferogram'
            target_array_str = f'HDF5:{input_runw}:/{src_iono_unct_path}'
            target_slc_array = isce3.io.Raster(target_array_str)
            rows_main = target_slc_array.length
            cols_main = target_slc_array.width

            if ('frequencyB' in src_h5[swath_path]):

                if 'listOfPolarizations' not in dst_h5[freq_path]:
                    h5_prep._add_polarization_list(dst_h5, 'RUNW',
                                                   CommonPaths().RootPath,
                                                   freq, pol)
                if 'interferogram' not in dst_h5[freq_path]:
                    dst_h5[freq_path].create_group('interferogram')

                if pol not in dst_h5[ifg_path]:
                    dst_h5[ifg_path].create_group(pol)

                if ('ionospherePhaseScreen' in src_h5[src_pol_path]) and \
                   ('ionospherePhaseScreen' not in dst_h5[dest_pol_path]):
                    iono_shape = src_h5[iono_path].shape
                    grids_val = 'None'
                    descr = "Split spectrum ionosphere phase screen"
                    h5_prep._create_datasets(
                        dst_h5[dest_pol_path],
                        iono_shape, np.float32,
                        'ionospherePhaseScreen',
                        descr=descr, units="radians",
                        grids=grids_val,
                        long_name='ionosphere phase screen')
                    descr = "Uncertainty of split spectrum ionosphere phase screen"
                    h5_prep._create_datasets(
                        dst_h5[dest_pol_path],
                        iono_shape, np.float32,
                        'ionospherePhaseScreenUncertainty',
                        descr=descr, units="radians",
                        grids=grids_val,
                        long_name='ionosphere phase \
                        screen uncertainty')
                nblocks = int(np.ceil(rows_main / blocksize))

                src_iono_paths = [src_iono_path, src_iono_unct_path]
                dst_iono_paths = [iono_path, iono_unct_path]
                for block in range(0, nblocks):
                    row_start = block * blocksize
                    if (row_start + blocksize > rows_main):
                        block_rows_data = rows_main - row_start
                    else:
                        block_rows_data = blocksize

                    for src_iono_path, dst_iono_path in zip(
                            src_iono_paths, dst_iono_paths):

                        iono = np.empty([block_rows_data, cols_main],
                                        dtype=float)
                        src_h5[src_iono_path].read_direct(
                            iono,
                            np.s_[row_start:row_start + block_rows_data, :])
                        if oversample_flag:
                            iono = interpolate_freq_b_array(slant_main,
                                                            slant_side,
                                                            iono)
                        dst_h5[dst_iono_path].write_direct(
                            iono,
                            dest_sel=np.s_[
                                    row_start:row_start+block_rows_data, :])

                # Add statistics to ionosphere datasets in RUNW
                for dst_iono_path in dst_iono_paths:
                    compute_stats_real_hdf5_dataset(dst_h5[dst_iono_path])


def compute_differential_phase(phase_first,
                               phase_second,
                               output_path,
                               first_data_path,
                               second_data_path,
                               output_data_path,
                               lines_per_block,
                               first_slant_path=None,
                               second_slant_path=None,
                               ):
    """
    Compute a differential phase by multiplying the first complex phase
    dataset with the complex conjugate of the second dataset, then write
    the result to an output HDF5.

    The function operates in blocks for memory efficiency. It can handle the
    case where `phase_first` and `output_path` refer to the same file (i.e.,
    in-place writing).

    Parameters
    ----------
    phase_first : str
        Path to the first HDF5 file containing the phase data.
    phase_second : str
        Path to the second HDF5 file containing the phase data.
    output_path : str
        Path to the HDF5 file where the differential phase result will be
        stored.
    first_data_path : list of str
        List of dataset paths within `phase_first`. Each path points to a
        raster where the first-phase data is stored.
    second_data_path : list of str
        List of dataset paths within `phase_second`. Each path points to a
         raster where the second-phase data is stored.
    output_data_path : list of str
        List of dataset paths within `output_path`. Each path points to a
        raster where the differential-phase output will be written.
    lines_per_block : int
        Number of lines (rows) to process per block for memory efficiency.
    first_slant_path : str, optional
        Dataset path in `phase_first` containing slant-range information (used
        for resampling). If `None`, no resampling is performed.
    second_slant_path : str, optional
        Dataset path in `phase_second` containing slant-range information (used
        for resampling). If `None`, no resampling is performed.

    Returns
    -------
    None
        The differential phase data is written directly to the specified
        output datasets. This function does not return any data.
    """
    def _to_complex_if_needed(arr):
        """
        Ensure an array is complex. If it's already complex, return as-is.
        Otherwise interpret values as phase (radians) and convert with exp(j*phi).
        """
        if np.iscomplexobj(arr):
            return arr
        # Cast to float to avoid integer exponentiation issues
        return np.exp(1j * arr)

    # Check if reading and writing will happen on the same file
    is_same_file_first_output = (phase_first == output_path)
    is_same_file_first_second = (phase_first == phase_second)

    # Open the relevant HDF5 files in appropriate modes
    with HDF5OptimizedReader(name=phase_first,
                             mode='r' if not is_same_file_first_output else 'a',
                             libver='latest',
                             swmr=True) as src_first_h5:
        src_sec_h5 = (src_first_h5 if is_same_file_first_second else
                      HDF5OptimizedReader(name=phase_second,
                                          mode='r',
                                          libver='latest',
                                          swmr=True))

        # If the first file is also the output file, reuse that handle.
        # Otherwise, open a new handle for the output file.
        src_out_h5 = (src_first_h5 if is_same_file_first_output else
                      HDF5OptimizedReader(name=output_path,
                                          mode='a',
                                          libver='latest',
                                          swmr=True))
        resampling_flag = False
        if first_slant_path is not None and second_slant_path is not None:
            main_slant = np.array(src_first_h5[first_slant_path])
            side_slant = np.array(src_sec_h5[second_slant_path])
            resampling_flag = True

        try:
            # Iterate over each polarization in frequency A
            for [first_ifg_path, second_ifg_path, out_ifg_path] in zip(
                 first_data_path, second_data_path, output_data_path):

                phase_first_raster = src_first_h5[first_ifg_path]
                phase_second_raster = src_sec_h5[second_ifg_path]
                output_data_raster = src_out_h5[out_ifg_path]

                # Generate block parameters for reading/writing
                block_params_main = block_param_generator(
                    lines_per_block,
                    [phase_first_raster.shape[0],
                     phase_first_raster.shape[1]],
                    [0, 0]
                )
                block_params_side = block_param_generator(
                    lines_per_block,
                    [phase_second_raster.shape[0],
                     phase_second_raster.shape[1]],
                    [0, 0]
                )

                # Process each block
                for block_param_main, block_param_side in zip(
                        block_params_main, block_params_side):
                    first_data_block = get_raster_block(phase_first_raster,
                                                        block_param_main)
                    if resampling_flag:
                        second_data_block = get_raster_block(
                            phase_second_raster,
                            block_param_side)
                        chosen_block_param = block_param_side
                    else:
                        second_data_block = get_raster_block(
                            phase_second_raster,
                            block_param_main)
                        chosen_block_param = block_param_main

                    # Ensure complex (convert real-valued phase in radians to
                    # complex phase)
                    first_data_block = _to_complex_if_needed(first_data_block)
                    second_data_block = _to_complex_if_needed(second_data_block)

                    # Optional resampling of FIRST to SECOND grid
                    if resampling_flag:
                        first_data_block = decimate_freq_a_array(
                            main_slant,
                            side_slant,
                            first_data_block)

                    # Compute the differential phase
                    diff_phase = first_data_block * np.conj(second_data_block)

                    # Write result block to output
                    write_raster_block(output_data_raster,
                                       diff_phase, chosen_block_param)
        finally:
            # Close the output file if it was opened separately
            if not is_same_file_first_second:
                src_sec_h5.close()
            if not is_same_file_first_output:
                src_out_h5.close()


def insar_ionosphere_pair(original_cfg, runw_hdf5):
    """Run insar workflow for additional interferogram to be used for
    ionosphere estimation

    - For split_main_band, upper and lower sub-bands interferograms are
    created.
    - For main_side_band and main_diff_ms_band, frequency B interferograms are
      created.

    If interferograms to be used for ionosphere estimation do not exist,
    they are generated by modifying original_cfg.
    For example, the frequency A and B interferograms are requested in VV
    polarization while ionosphere is estimated in HH polarization, the additional
    HH interferogram is created.

    Parameters
    ----------
    original_cfg : dict
        dictionary of runconfigs
    runw_hdf5: str
        File path to runw HDF5 product (i.e., RUNW)
    """

    # ionosphere runconfigs
    iono_args = original_cfg['processing']['ionosphere_phase_correction']
    scratch_path = original_cfg['product_path_group']['scratch_path']

    # pull parameters for ionosphere phase estimation
    iono_freq_pols = iono_args['list_of_frequencies']
    iono_method = iono_args['spectral_diversity']

    iono_path = os.path.join(scratch_path, 'ionosphere')
    split_slc_path = os.path.join(iono_path, 'split_spectrum')

    _, original_out_paths = h5_prep.get_products_and_paths(original_cfg)

    # Keep original_cfg before changing it
    partial_orig_cfg_dict = dict()
    partial_orig_cfg_dict['scratch_path'] = scratch_path
    partial_orig_cfg_dict['reference_rslc_file'] = \
        original_cfg['input_file_group']['reference_rslc_file']
    partial_orig_cfg_dict['secondary_rslc_file'] = \
        original_cfg['input_file_group']['secondary_rslc_file']
    partial_orig_cfg_dict['coregistered_slc_path'] = \
        original_cfg['processing']['crossmul']['coregistered_slc_path']
    partial_orig_cfg_dict['list_of_frequencies'] = \
        original_cfg['processing']['input_subset']['list_of_frequencies']
    partial_orig_cfg_dict['output_runw'] = runw_hdf5
    if original_cfg['processing']['fine_resample']['enabled']:

        resample_type = 'fine'
    else:
        resample_type = 'coarse'

    partial_orig_cfg_dict['offsets_dir'] = original_cfg['processing'][
        f'{resample_type}_resample']['offsets_dir']

    orig_scratch_path = scratch_path
    orig_freq_pols = copy.deepcopy(original_cfg['processing']['input_subset'][
                    'list_of_frequencies'])
    orig_product_type = original_cfg['primary_executable']['product_type']

    iono_insar_cfg = deepcopy_runconfig_and_keep_isce3_obj(original_cfg)
    iono_insar_cfg['primary_executable'][
                'product_type'] = 'RUNW'

    # It is sufficient to compute crossmul once at 80 m posting for computing
    # the ionosphere. Therefore, we switch off the number of looks from
    # crossmul (30 m) with that of unwrapping (80 m)

    runw_rg_looks = iono_insar_cfg[
        'processing']['phase_unwrap']['range_looks']
    runw_az_looks = iono_insar_cfg[
        'processing']['phase_unwrap']['azimuth_looks']

    if runw_rg_looks != 1 or runw_az_looks != 1:
        iono_insar_cfg[
            'processing']['crossmul']['range_looks'] = runw_rg_looks
        iono_insar_cfg[
            'processing']['crossmul']['azimuth_looks'] = runw_az_looks

        iono_insar_cfg['processing']['phase_unwrap']['range_looks'] = 1
        iono_insar_cfg['processing']['phase_unwrap']['azimuth_looks'] = 1

    iono_unwrapped_cfg = iono_insar_cfg[
        'processing']['phase_unwrap']

    if iono_unwrapped_cfg['range_looks'] > 1 and \
       iono_unwrapped_cfg['azimuth_looks'] > 1:

        iono_insar_cfg['processing']['crossmul']['range_looks'] = \
            iono_unwrapped_cfg['range_looks']
        iono_insar_cfg['processing']['crossmul']['azimuth_looks'] = \
            iono_unwrapped_cfg['azimuth_looks']
        iono_unwrapped_cfg['range_looks'] = 1
        iono_unwrapped_cfg['azimuth_looks'] = 1

    # update processing parameter
    # water mask for ionosphere is not supported now.
    prep_wrapped_phase_cfg = iono_insar_cfg['processing'][
        'phase_unwrap']['preprocess_wrapped_phase']
    unwrap_mask_type = prep_wrapped_phase_cfg['mask']['mask_type']

    if unwrap_mask_type == 'water':
        # Either set to a default value or delete the key entirely.
        prep_wrapped_phase_cfg['enabled'] = True

    if iono_method in ['split_main_band', 'main_diff_low_high_subband']:
        # For split_main_band, two sub-band interferograms need to be
        # created
        for split_str in ['low', 'high']:

            # update reference sub-band path
            ref_h5_path = os.path.join(split_slc_path,
                                       f"ref_{split_str}_band_slc.h5")
            iono_insar_cfg['input_file_group'][
                'reference_rslc_file'] = ref_h5_path

            # update secondary sub-band path
            sec_h5_path = os.path.join(split_slc_path,
                                       f"sec_{split_str}_band_slc.h5")
            iono_insar_cfg['input_file_group'][
                'secondary_rslc_file'] = sec_h5_path

            # update output path
            new_scratch = pathlib.Path(orig_scratch_path,
                                       'ionosphere', split_str)
            iono_insar_cfg['product_path_group'][
                'scratch_path'] = new_scratch
            iono_insar_cfg['product_path_group'][
                'sas_output_file'] = f'{new_scratch}/RUNW.h5'
            iono_insar_cfg['processing']['dense_offsets'][
                'coregistered_slc_path'] = new_scratch
            iono_insar_cfg['processing']['crossmul'][
                'coregistered_slc_path'] = new_scratch

            # update frequency and polarizations for ionosphere
            if iono_freq_pols['A']:
                iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies']['A'] = iono_freq_pols['A']
            if iono_freq_pols['B']:
                iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies']['B'] = iono_freq_pols['B']
            else:
                # if cfg has key for frequency B, then delete it to avoid
                # unnecessary insar processing
                try:
                    del iono_insar_cfg['processing']['input_subset'][
                        'list_of_frequencies']['B']
                except:
                    pass

            # create directory for sub-band interferograms
            new_scratch.mkdir(parents=True, exist_ok=True)

            # run insar for sub-band SLCs
            _, out_paths = h5_prep.get_products_and_paths(iono_insar_cfg)
            out_paths['RUNW'] = f'{new_scratch}/RUNW.h5'

            if iono_method in ['main_diff_low_high_subband']:
                unwrapping_flag = False
            else:
                unwrapping_flag = True

            run_insar_workflow(iono_insar_cfg,
                               partial_orig_cfg_dict,
                               out_paths,
                               unwrapping_flag=unwrapping_flag)

            if iono_method in ['main_diff_low_high_subband'] and \
                    split_str == 'high':
                diff_dir = pathlib.Path(orig_scratch_path,
                                        'ionosphere', 'diff_low_high')
                phase_first = pathlib.Path(orig_scratch_path,
                                           'ionosphere', 'high', 'RIFG.h5')
                phase_second = pathlib.Path(orig_scratch_path,
                                            'ionosphere', 'low', 'RIFG.h5')
                diff_phase_output = pathlib.Path(diff_dir, 'RIFG.h5')
                unwrapped_phase_first = pathlib.Path(
                    orig_scratch_path, 'ionosphere', 'high', 'RUNW.h5')
                diff_unwrapped_phase_output = pathlib.Path(diff_dir, 'RUNW.h5')
                iono_insar_cfg['product_path_group'][
                    'scratch_path'] = diff_dir
                iono_insar_cfg['product_path_group'][
                    'sas_output_file'] = f'{diff_dir}/RUNW.h5'
                iono_insar_cfg['processing']['dense_offsets'][
                    'coregistered_slc_path'] = new_scratch
                iono_insar_cfg['processing']['crossmul'][
                    'coregistered_slc_path'] = new_scratch
                _, out_paths = h5_prep.get_products_and_paths(iono_insar_cfg)
                os.makedirs(diff_dir, exist_ok=True)
                shutil.copy(phase_first, diff_phase_output)
                shutil.copy(unwrapped_phase_first, diff_unwrapped_phase_output)

                sym_iono_rdr2geo_dir = os.path.abspath(
                    f"{diff_dir}/rdr2geo")

                if not os.path.lexists(sym_iono_rdr2geo_dir):
                    os.symlink(
                        os.path.abspath(f"{scratch_path}/rdr2geo"),
                        sym_iono_rdr2geo_dir,
                        target_is_directory=True)

                pol_list_a = iono_freq_pols['A']
                swath_path = RIFGGroupsPaths().SwathsPath
                first_data_path = []
                for pol_a in pol_list_a:

                    dest_freq_path = f"{swath_path}/frequencyA"
                    dest_pol_path = f"{dest_freq_path}/interferogram/{pol_a}"
                    rifg_path_freq = f"{dest_pol_path}/wrappedInterferogram"

                    first_data_path.append(rifg_path_freq)
                second_data_path = first_data_path
                output_data_path = first_data_path
                compute_differential_phase(phase_first,
                                           phase_second,
                                           diff_phase_output,
                                           first_data_path,
                                           second_data_path,
                                           output_data_path,
                                           iono_args['lines_per_block'])
                # Since main_diff_low_high_subband method does not need to
                # unwrap low and high subband interferogram, but need to
                # unwrap the difference between low and high subband
                # interferogram
                unwrap.run(iono_insar_cfg, out_paths['RIFG'], out_paths['RUNW'])

    elif iono_method in ['main_side_band', 'main_diff_ms_band']:
        rerun_insar_pairs = 0
        for freq in iono_freq_pols.keys():
            iono_pol = iono_freq_pols[freq]
            try:
                orig_pol = orig_freq_pols[freq]
            except:
                orig_pol = []
            res_pol = [pol for pol in iono_pol if pol not in orig_pol]
            # update frequency and polarizations for ionosphere
            if res_pol:
                iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies'][freq] = res_pol
                rerun_insar_pairs += 1
            else:
                del iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies'][freq]

        if iono_method in ['main_diff_ms_band']:
            unwrapping_flag = False
        else:
            unwrapping_flag = True

        if rerun_insar_pairs > 0:
            # update paths
            new_scratch = pathlib.Path(iono_path, f'{iono_method}')
            iono_insar_cfg['product_path_group'][
                'scratch_path'] = new_scratch
            iono_insar_cfg['processing']['geo2rdr'][
                'topo_path'] = new_scratch
            iono_insar_cfg['product_path_group'][
                'sas_output_file'] = f'{new_scratch}/RUNW.h5'
            iono_insar_cfg['processing']['dense_offsets'][
                'coregistered_slc_path'] = new_scratch
            iono_insar_cfg['processing']['crossmul'][
                'coregistered_slc_path'] = new_scratch

            new_scratch.mkdir(parents=True, exist_ok=True)

            _, out_paths = h5_prep.get_products_and_paths(iono_insar_cfg)
            out_paths['RUNW'] = f'{new_scratch}/RUNW.h5'
            additional_runw = f'{new_scratch}/RUNW.h5'
            run_insar_workflow(iono_insar_cfg,
                               partial_orig_cfg_dict,
                               out_paths,
                               unwrapping_flag=unwrapping_flag)

        if iono_method == 'main_diff_ms_band':
            diff_dir = pathlib.Path(orig_scratch_path,
                                    'ionosphere', 'diff_ms')
            phase_first = original_out_paths['RUNW']
            if rerun_insar_pairs > 0:
                additional_runw = f'{new_scratch}/RUNW.h5'
                phase_second = pathlib.Path(orig_scratch_path,
                                            'ionosphere',
                                            'main_diff_ms_band',
                                            'RIFG.h5')
            else:
                out_paths = original_out_paths
                new_scratch = orig_scratch_path
                phase_second = out_paths['RIFG']
                additional_runw = out_paths['RUNW']

            diff_phase_output = pathlib.Path(diff_dir, 'RIFG.h5')
            iono_insar_cfg['product_path_group'][
                'scratch_path'] = diff_dir
            iono_insar_cfg['product_path_group'][
                'sas_output_file'] = f'{diff_dir}/RUNW.h5'
            iono_insar_cfg['processing']['dense_offsets'][
                'coregistered_slc_path'] = new_scratch
            iono_insar_cfg['processing']['crossmul'][
                'coregistered_slc_path'] = new_scratch
            iono_insar_cfg['processing']['input_subset'][
                'list_of_frequencies']['B'] = iono_pol
            iono_insar_cfg['processing']['input_subset'][
                'list_of_frequencies']['A'] = []
            _, out_paths = h5_prep.get_products_and_paths(iono_insar_cfg)
            os.makedirs(diff_dir, exist_ok=True)
            shutil.copy(phase_second, diff_phase_output)
            shutil.copy(additional_runw, out_paths['RUNW'])

            pol_list_a = iono_freq_pols['A']
            pol_list_b = iono_freq_pols['B']
            swath_path = RIFGGroupsPaths().SwathsPath
            runw_swath_path = RUNWGroupsPaths().SwathsPath

            first_data_path = []
            for pol_a in pol_list_a:

                dest_freq_path = f"{runw_swath_path}/frequencyA"
                dest_pol_path = f"{dest_freq_path}/interferogram/{pol_a}"
                runw_path_freq = f"{dest_pol_path}/unwrappedPhase"

                first_data_path.append(runw_path_freq)
            first_slant_path = f"{dest_freq_path}/interferogram/slantRange"
            second_data_path = []
            for pol_b in pol_list_b:

                dest_freq_path = f"{swath_path}/frequencyB"
                dest_pol_path = f"{dest_freq_path}/interferogram/{pol_b}"
                rifg_path_freq = f"{dest_pol_path}/wrappedInterferogram"

                second_data_path.append(rifg_path_freq)
            second_slant_path = f"{dest_freq_path}/interferogram/slantRange"

            output_data_path = second_data_path
            compute_differential_phase(phase_first,
                                       phase_second,
                                       diff_phase_output,
                                       first_data_path,
                                       second_data_path,
                                       output_data_path,
                                       iono_args['lines_per_block'],
                                       first_slant_path=first_slant_path,
                                       second_slant_path=second_slant_path)

            # Since main_diff_low_high_subband method does not need to
            # unwrap low and high subband interferogram, but need to
            # unwrap the difference between low and high subband interferogram
            unwrap.run(iono_insar_cfg, out_paths['RIFG'], out_paths['RUNW'])

    # restore original paths
    original_cfg['input_file_group']['reference_rslc_file'] = \
        partial_orig_cfg_dict['reference_rslc_file']
    original_cfg['input_file_group']['secondary_rslc_file'] = \
        partial_orig_cfg_dict['secondary_rslc_file']
    original_cfg['product_path_group']['scratch_path'] = \
        partial_orig_cfg_dict['scratch_path']
    original_cfg['processing']['dense_offsets']['coregistered_slc_path'] = \
        partial_orig_cfg_dict['coregistered_slc_path']
    original_cfg['processing']['crossmul']['coregistered_slc_path'] = \
        partial_orig_cfg_dict['coregistered_slc_path']

    original_cfg['processing']['input_subset'][

            'list_of_frequencies'] = orig_freq_pols
    original_cfg['primary_executable'][
                'product_type'] = orig_product_type
    original_cfg['processing']['geo2rdr']['topo_path'] = orig_scratch_path


def run_insar_workflow(iono_insar_cfg, original_dict, out_paths,
                       unwrapping_flag=True):
    '''Run InSAR workflow for ionosphere estimation pair without
    rdr2geo and geo2rdr steps

    - For split_main_band, rdr2geo and geo2rdr computed from insar workflow
      are used.
    - For methods using side-band, offsets are decimated and used for
    frequency B interferogram generation.

    Parameters
    ---------
    iono_insar_cfg: dict
        InSAR workflow runconfig dictionary modified with ionosphere pairs
        and ionosphere specific conditions
    original_dict: dict
        dictionary containing following parameters
        from original InSAR runconfig
        - scratch_path
        - reference_rslc_file
        - secondary_rslc_file
        - coregistered_slc_path
        - list_of_frequencies
        - output_runw
        - offsets_dir
    out_paths: dict
        output files (RIFG, RUNW)for out_paths
    unwrapping_flag : boolean
        flag indicating whether to unwrap the output
    '''

    # run insar for ionosphere pairs
    prepare_insar_hdf5.run(iono_insar_cfg)

    # create symbolic links for rdr2geo
    sym_iono_rdr2geo_dir = os.path.abspath(
        f"{iono_insar_cfg['product_path_group']['scratch_path']}/rdr2geo")

    if not os.path.lexists(sym_iono_rdr2geo_dir):
        os.symlink(
            os.path.abspath(f"{original_dict['scratch_path']}/rdr2geo"),
            sym_iono_rdr2geo_dir,
            target_is_directory=True)

    iono_freq_pol = iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies']
    # decimate offsets for frequency B and create ionosphere layers
    if 'B' in iono_freq_pol:
        decimate_freq_a_offset(iono_insar_cfg, original_dict)

    if iono_insar_cfg['processing']['fine_resample']['enabled']:
        resample_slc_v2.run(iono_insar_cfg, 'fine')
    else:
        resample_slc_v2.run(iono_insar_cfg, 'coarse')

    if iono_insar_cfg['processing']['fine_resample']['enabled']:
        crossmul.run(iono_insar_cfg, out_paths['RIFG'], 'fine')
    else:
        crossmul.run(iono_insar_cfg, out_paths['RIFG'], 'coarse')

    if iono_insar_cfg['processing']['filter_interferogram']['filter_type'] != 'no_filter':
        filter_interferogram.run(iono_insar_cfg, out_paths['RIFG'])

    if 'RUNW' in out_paths and unwrapping_flag:
        unwrap.run(iono_insar_cfg, out_paths['RIFG'], out_paths['RUNW'])


def run(cfg: dict, runw_hdf5: str):
    '''
    Run ionosphere phase correction workflow with parameters
    in cfg dictionary
    Parameters
    ---------
    cfg: dict
        Dictionary with user-defined options
    runw_hdf5: str
        File path to runw HDF5 product (i.e., RUNW)
    '''

    # Create error and info channels
    info_channel = journal.info("ionosphere_phase_correction.run")
    info_channel.log("starting insar_ionosphere_correction")

    # Instantiate RUNW object to easy access RUNW datasets
    runw_obj = RUNWGroupsPaths()

    # pull parameters from dictionary
    iono_args = cfg['processing']['ionosphere_phase_correction']
    scratch_path = cfg['product_path_group']['scratch_path']

    # pull parameters for ionosphere phase estimation
    iono_freq_pols = copy.deepcopy(iono_args['list_of_frequencies'])
    iono_method = iono_args['spectral_diversity']
    blocksize = iono_args['lines_per_block']
    filter_cfg = iono_args['dispersive_filter']

    # pull parameters for dispersive filter
    filter_bool = filter_cfg['enabled']
    mask_type = filter_cfg['filter_mask_type']
    filter_coh_thresh = filter_cfg['filter_coherence_threshold']
    kernel_range_size = filter_cfg['kernel_range']
    kernel_azimuth_size = filter_cfg['kernel_azimuth']
    kernel_sigma_range = filter_cfg['sigma_range']
    kernel_sigma_azimuth = filter_cfg['sigma_azimuth']
    filling_method = filter_cfg['filling_method']
    filter_iterations = filter_cfg['filter_iterations']
    median_filter_size = filter_cfg['median_filter_size']
    median_filter_threshold = filter_cfg['median_filter_threshold']
    min_cluster_pixels = filter_cfg['min_cluster_pixels']
    unwrap_correction_bool = filter_cfg['unwrap_correction']

    # bridge algorithm options
    bridge_algorithm_bool = filter_cfg['bridge_algorithm_enabled']
    bridge_minimum_samples = filter_cfg['bridge_minimum_samples']
    bridge_radius = filter_cfg['bridge_radius']
    bridge_erosion_size = filter_cfg['bridge_erosion_size']
    bridge_deramp_type = filter_cfg['bridge_ramp_type']
    bridge_ramp_maximum_pixel = filter_cfg['bridge_ramp_maximum_pixel']

    rg_looks = cfg['processing']['crossmul']['range_looks']
    az_looks = cfg['processing']['crossmul']['azimuth_looks']
    unwrap_rg_looks = cfg['processing']['phase_unwrap']['range_looks']
    unwrap_az_looks = cfg['processing']['phase_unwrap']['azimuth_looks']

    if unwrap_rg_looks != 1 or unwrap_az_looks != 1:
        rg_looks = unwrap_rg_looks
        az_looks = unwrap_az_looks

    # set paths for ionosphere and split spectrum
    iono_path = os.path.join(scratch_path, 'ionosphere')
    split_slc_path = os.path.join(iono_path, 'split_spectrum')

    # Keep cfg before changing it
    orig_scratch_path = cfg['product_path_group']['scratch_path']
    orig_ref_str = cfg['input_file_group']['reference_rslc_file']
    orig_sec_str = cfg['input_file_group']['secondary_rslc_file']
    orig_freq_pols = copy.deepcopy(cfg['processing']['input_subset'][
                    'list_of_frequencies'])
    iono_insar_cfg = deepcopy_runconfig_and_keep_isce3_obj(cfg)

    # Run InSAR for sub-band SLCs (split-main-bands) or
    # for main and side bands for iono_freq_pols (main-side-bands)
    insar_ionosphere_pair(iono_insar_cfg, runw_hdf5)

    t_all = time.time()
    # Define methods to use subband or sideband
    iono_method_subbands = ['split_main_band', 'main_diff_low_high_subband']
    iono_method_sideband = ['main_side_band', 'main_diff_ms_band']

    # set frequency A RUNW path
    if runw_hdf5:
        runw_path_insar = runw_hdf5
    else:
        runw_path_insar = os.path.join(scratch_path, 'RUNW.h5')

    # Start ionosphere phase estimation
    # pull center frequency from frequency A, which is used for all method
    base_ref_slc_str = orig_ref_str
    base_ref_slc = SLC(hdf5file=base_ref_slc_str)
    ref_meta_data_a = splitspectrum.BandpassMetaData.load_from_slc(
        slc_product=base_ref_slc,
        freq='A')
    f0 = ref_meta_data_a.center_freq

    if iono_method in iono_method_subbands:
        # pull center frequencies from sub-bands
        high_ref_slc_str = os.path.join(split_slc_path, "ref_high_band_slc.h5")
        low_ref_slc_str = os.path.join(split_slc_path, "ref_low_band_slc.h5")
        high_ref_slc = SLC(hdf5file=high_ref_slc_str)
        low_ref_slc = SLC(hdf5file=low_ref_slc_str)

        high_sub_meta_data = splitspectrum.BandpassMetaData.load_from_slc(
            slc_product=high_ref_slc,
            freq='A')
        low_sub_meta_data = splitspectrum.BandpassMetaData.load_from_slc(
            slc_product=low_ref_slc,
            freq='A')
        f0_low = low_sub_meta_data.center_freq
        f0_high = high_sub_meta_data.center_freq

        f1 = None

        if iono_method == "split_main_band":
            IonosphereEstimationMethod = LowHighSubbandIonosphereEstimation
        elif iono_method == "main_diff_low_high_subband":
            IonosphereEstimationMethod = \
                MainDiffLowHighSubbandIonosphereEstimation

    if iono_method in iono_method_sideband:
        # pull center frequency from frequency B
        ref_meta_data_b = splitspectrum.BandpassMetaData.load_from_slc(
            slc_product=base_ref_slc,
            freq='B')
        f1 = ref_meta_data_b.center_freq

        # find polarizations which are not processed in InSAR workflow
        if 'A' in orig_freq_pols:
            residual_pol_a = list(set(
                iono_freq_pols['A']) - set(orig_freq_pols['A']))
        else:
            residual_pol_a = list(iono_freq_pols['A'])

        if 'B' in orig_freq_pols:
            residual_pol_b = list(set(
                iono_freq_pols['B']) - set(orig_freq_pols['B']))
        else:
            residual_pol_b = list(iono_freq_pols['B'])
        f0_low = None
        f0_high = None

        if iono_method == "main_side_band":
            IonosphereEstimationMethod = MainSideBandIonosphereEstimation
        else:
            IonosphereEstimationMethod = MainDiffMsBandIonosphereEstimation

    # Create object for ionosphere esimation
    iono_phase_obj = IonosphereEstimationMethod(
        main_center_freq=f0,
        side_center_freq=f1,
        low_center_freq=f0_low,
        high_center_freq=f0_high)

    # Create object for ionosphere filter
    iono_filter_obj = IonosphereFilter(
        x_kernel=kernel_range_size,
        y_kernel=kernel_azimuth_size,
        sig_x=kernel_sigma_range,
        sig_y=kernel_sigma_azimuth,
        iteration=filter_iterations,
        filling_method=filling_method,
        outputdir=os.path.join(iono_path, iono_method))

    # pull parameters for polarizations
    pol_list_a = list(iono_freq_pols['A'])
    if iono_method in iono_method_sideband:
        pol_list_b = list(iono_freq_pols['B'])
    # Read Block and estimate dispersive and non-dispersive
    for pol_ind, pol_a in enumerate(pol_list_a):

        # Set paths for upwrapped interferogram, coherence,
        # connected components and slant range
        iono_output = runw_path_insar
        pol_comb_str = f"{pol_a}_{pol_a}"
        swath_path = runw_obj.SwathsPath
        dest_freq_path = f"{swath_path}/frequencyA"
        dest_pol_path = f"{dest_freq_path}/interferogram/{pol_a}"
        output_pol_path = dest_pol_path

        runw_path_freq_a = f"{dest_pol_path}/unwrappedPhase"
        rcoh_path_freq_a = f"{dest_pol_path}/coherenceMagnitude"
        rcom_path_freq_a = f"{dest_pol_path}/connectedComponents"
        rslant_path_a = f"{dest_freq_path}/interferogram/"\
            "slantRange"
        subswath_mask_freq_a_path = f"{dest_freq_path}/interferogram/mask"

        # Set paths for frequency B
        if iono_method in iono_method_sideband:
            pol_b = pol_list_b[pol_ind]
            pol_comb_str = f"{pol_a}_{pol_b}"
            dest_freq_path_b = f"{swath_path}/frequencyB"
            dest_pol_path_b = f"{dest_freq_path_b}/interferogram/{pol_b}"
            output_pol_path = f"{dest_freq_path_b}/interferogram/{pol_a}"
            runw_path_freq_b = f"{dest_pol_path_b}/unwrappedPhase"
            rcoh_path_freq_b = f"{dest_pol_path_b}/coherenceMagnitude"
            rcom_path_freq_b = f"{dest_pol_path_b}/connectedComponents"
            rslant_path_b = f"{dest_freq_path_b}/interferogram/"\
                "slantRange"
            subswath_mask_freq_b_path = \
                f"{dest_freq_path_b}/interferogram/mask"

        if iono_method in iono_method_subbands:
            # set paths for high and low sub-bands
            sub_low_runw_str = os.path.join(iono_path, 'low', 'RUNW.h5')
            sub_high_runw_str = os.path.join(iono_path, 'high', 'RUNW.h5')
            sub_diff_runw_str = os.path.join(iono_path,
                                             'diff_low_high',
                                             'RUNW.h5')

            target_array_str = f'HDF5:{sub_low_runw_str}:/{runw_path_freq_a}'
            target_slc_array = isce3.io.Raster(target_array_str)
            rows_main = target_slc_array.length
            cols_main = target_slc_array.width
            nblocks = int(np.ceil(rows_main / blocksize))
            rows_output = rows_main
            cols_output = cols_main
            # In method using only sub-bands, resampling is unnecessary.
            # thus, slant range info is not needed.
            main_slant = None
            side_slant = None

        if iono_method in iono_method_sideband:
            # set paths for HDF5 that have frequency A unwrapped phase
            if pol_a in residual_pol_a:
                runw_freq_a_str = os.path.join(
                    iono_path, iono_method, 'RUNW.h5')
            # If target polarization is in pre-existing HDF5,
            # then use it without additional InSAR workflow.
            else:
                runw_freq_a_str = runw_path_insar
            # set paths for HDF5 that have frequency B unwrapped phase
            if pol_b in residual_pol_b:
                runw_freq_b_str = os.path.join(iono_path,
                                               iono_method,
                                               'RUNW.h5')
            else:
                runw_freq_b_str = runw_path_insar

            runw_diff_str = os.path.join(iono_path,
                                         'diff_ms',
                                         'RUNW.h5')
            iono_output = runw_freq_b_str
            iono_output_runw = runw_path_insar

            main_raster_str = f'HDF5:{runw_freq_a_str}:/{runw_path_freq_a}'
            main_runw_raster = isce3.io.Raster(main_raster_str)
            rows_main = main_runw_raster.length
            cols_main = main_runw_raster.width
            nblocks = int(np.ceil(rows_main / blocksize))

            side_raster_str = f'HDF5:{runw_freq_b_str}:/{runw_path_freq_b}'
            side_runw_raster = isce3.io.Raster(side_raster_str)
            rows_side = side_runw_raster.length
            cols_side = side_runw_raster.width

            main_slant = np.empty([cols_main], dtype=float)
            side_slant = np.empty([cols_side], dtype=float)
            rows_output = rows_side
            cols_output = cols_side
            del main_runw_raster
            del side_runw_raster

            with HDF5OptimizedReader(
                    name=runw_freq_a_str, mode='r',
                    libver='latest', swmr=True) as src_main_h5, \
                HDF5OptimizedReader(
                    name=runw_freq_b_str, mode='r',
                    libver='latest', swmr=True) as src_side_h5:

                # Read slant range block from HDF5
                src_main_h5[rslant_path_a].read_direct(
                    main_slant, np.s_[:])
                src_side_h5[rslant_path_b].read_direct(
                    side_slant, np.s_[:])

        for block in range(0, nblocks):
            info_channel.log(f"Ionosphere Phase Estimation block: {block}")

            row_start = block * blocksize
            if (row_start + blocksize > rows_main):
                block_rows_data = rows_main - row_start
            else:
                block_rows_data = blocksize

            # initialize arrays by setting None
            sub_low_image = None
            sub_high_image = None
            main_image = None
            side_image = None
            diff_ms_image = None
            diff_subband_image = None

            sub_low_coh_image = None
            sub_high_coh_image = None
            main_coh_image = None
            side_coh_image = None
            diff_coh_image = None
            diff_ms_coh_image = None

            sub_low_conn_image = None
            sub_high_conn_image = None
            main_conn_image = None
            side_conn_image = None
            diff_ms_conn_image = None

            # ionosphere method using sub-bands uses subswath mask in freq A
            # ionosphere method using side-band uses subswath masks in freq A and B
            subswath_mask_image = None
            subswath_mask_main_image = None
            subswath_mask_side_image = None

            if iono_method in iono_method_subbands:
                # Initialize array for block rasters
                sub_low_image = np.empty([block_rows_data, cols_main],
                                         dtype=float)
                sub_high_image = np.empty([block_rows_data, cols_main],
                                          dtype=float)
                sub_low_coh_image = np.empty([block_rows_data, cols_main],
                                             dtype=float)
                sub_high_coh_image = np.empty([block_rows_data, cols_main],
                                              dtype=float)

                if "connected_components" in mask_type:
                    sub_low_conn_image = np.empty(
                        [block_rows_data, cols_main],
                        dtype=float)
                    sub_high_conn_image = np.empty(
                        [block_rows_data, cols_main],
                        dtype=float)

                if "subswath_mask" in mask_type:
                    subswath_mask_image = np.empty(
                        [block_rows_data, cols_main],
                        dtype=int)

                if iono_method == 'main_diff_low_high_subband':
                    main_image = np.empty([block_rows_data, cols_main],
                                          dtype=float)
                    diff_subband_image = np.empty([block_rows_data, cols_main],
                                                  dtype=float)
                    main_coh_image = np.empty([block_rows_data, cols_main],
                                              dtype=float)
                    diff_coh_image = np.empty([block_rows_data, cols_main],
                                              dtype=float)

                    if "connected_components" in mask_type:
                        main_conn_image = np.empty(
                            [block_rows_data, cols_main],
                            dtype=float)
                        diff_subband_conn_image = np.empty(
                            [block_rows_data, cols_main],
                            dtype=float)

                with HDF5OptimizedReader(
                        name=sub_low_runw_str, mode='r',
                        libver='latest', swmr=True) as src_low_h5, \
                    HDF5OptimizedReader(
                        name=sub_high_runw_str, mode='r',
                        libver='latest', swmr=True) as src_high_h5:

                    # Read runw block for sub-bands
                    src_low_h5[runw_path_freq_a].read_direct(
                        sub_low_image,
                        np.s_[row_start:row_start + block_rows_data, :])
                    src_high_h5[runw_path_freq_a].read_direct(
                        sub_high_image,
                        np.s_[row_start:row_start + block_rows_data, :])
                    # Read coherence block for sub-bands
                    src_low_h5[rcoh_path_freq_a].read_direct(
                        sub_low_coh_image,
                        np.s_[row_start:row_start + block_rows_data, :])
                    src_high_h5[rcoh_path_freq_a].read_direct(
                        sub_high_coh_image,
                        np.s_[row_start:row_start + block_rows_data, :])

                    if "connected_components" in mask_type:
                        # Read connected_components block for sub-bands
                        src_low_h5[rcom_path_freq_a].read_direct(
                            sub_low_conn_image,
                            np.s_[row_start:row_start + block_rows_data, :])
                        src_high_h5[rcom_path_freq_a].read_direct(
                            sub_high_conn_image,
                            np.s_[row_start:row_start + block_rows_data, :])

                    if "subswath_mask" in mask_type:
                        src_low_h5[subswath_mask_freq_a_path].read_direct(
                            subswath_mask_image,
                            np.s_[row_start:row_start + block_rows_data, :])

                if bridge_algorithm_bool:
                    sub_high_image = bridge_unwrapped_phase(
                        sub_high_image,
                        radius=bridge_radius,
                        min_num_pixel=bridge_minimum_samples,
                        erosion_size=bridge_erosion_size,
                        ramp_type=bridge_deramp_type,
                        deramp_max_num_sample=bridge_ramp_maximum_pixel)
                    sub_low_image = bridge_unwrapped_phase(
                        sub_low_image,
                        radius=bridge_radius,
                        min_num_pixel=bridge_minimum_samples,
                        erosion_size=bridge_erosion_size,
                        ramp_type=bridge_deramp_type,
                        deramp_max_num_sample=bridge_ramp_maximum_pixel)

                if unwrap_correction_bool:
                    sub_high_image = unwrapping_correction_with_filter(
                        sub_high_image,
                        kernel_width=kernel_range_size,
                        kernel_length=kernel_azimuth_size,
                        sig_kernel_x=kernel_sigma_range,
                        sig_kernel_y=kernel_sigma_azimuth,
                        iterations=filter_iterations,
                        filter_method='convolution')
                    sub_low_image = unwrapping_correction_with_filter(
                        sub_low_image,
                        kernel_width=kernel_range_size,
                        kernel_length=kernel_azimuth_size,
                        sig_kernel_x=kernel_sigma_range,
                        sig_kernel_y=kernel_sigma_azimuth,
                        iterations=filter_iterations,
                        filter_method='convolution')

                if iono_method == 'main_diff_low_high_subband':
                    with HDF5OptimizedReader(
                            name=runw_path_insar, mode='r',
                            libver='latest', swmr=True) as src_main_h5, \
                        HDF5OptimizedReader(
                            name=sub_diff_runw_str, mode='r',
                            libver='latest', swmr=True) as src_diff_h5:
                        src_main_h5[runw_path_freq_a].read_direct(
                            main_image,
                            np.s_[row_start:row_start + block_rows_data,
                                  :])
                        src_diff_h5[runw_path_freq_a].read_direct(
                            diff_subband_image,
                            np.s_[row_start:row_start + block_rows_data,
                                  :])

                        # Read coherence block for sub-bands
                        src_main_h5[rcoh_path_freq_a].read_direct(
                            main_coh_image,
                            np.s_[row_start:row_start + block_rows_data,
                                  :]
                            )
                        src_diff_h5[rcoh_path_freq_a].read_direct(
                            diff_coh_image,
                            np.s_[row_start:row_start + block_rows_data,
                                  :]
                            )

                        if "connected_components" in mask_type:
                            # Read connected_components block for sub-bands
                            src_main_h5[rcom_path_freq_a].read_direct(
                                main_conn_image,
                                np.s_[
                                    row_start:row_start + block_rows_data,
                                    :])
                            src_high_h5[rcom_path_freq_a].read_direct(
                                diff_subband_conn_image,
                                np.s_[
                                    row_start:row_start + block_rows_data,
                                    :])

            if iono_method in iono_method_sideband:

                main_image = np.empty([block_rows_data, cols_main],
                                      dtype=float)
                side_image = np.empty([block_rows_data, cols_side],
                                      dtype=float)

                main_coh_image = np.empty([block_rows_data, cols_main],
                                          dtype=float)
                side_coh_image = np.empty([block_rows_data, cols_side],
                                          dtype=float)
                if iono_method == 'main_diff_ms_band':
                    diff_ms_image = np.empty([block_rows_data, cols_side],
                                             dtype=float)
                    diff_ms_coh_image = np.empty([block_rows_data, cols_side],
                                                 dtype=float)

                if "connected_components" in mask_type:
                    main_conn_image = np.empty(
                        [block_rows_data, cols_main],
                        dtype=float)
                    side_conn_image = np.empty(
                        [block_rows_data, cols_side],
                        dtype=float)
                    diff_ms_conn_image = np.empty(
                        [block_rows_data, cols_side],
                        dtype=float)

                if "subswath_mask" in mask_type:
                    subswath_mask_main_image = np.empty(
                        [block_rows_data, cols_main],
                        dtype=int)
                    subswath_mask_side_image = np.empty(
                        [block_rows_data, cols_side],
                        dtype=int)

                with HDF5OptimizedReader(
                        name=runw_freq_a_str, mode='r',
                        libver='latest', swmr=True) as src_main_h5, \
                    HDF5OptimizedReader(
                        name=runw_freq_b_str, mode='r',
                        libver='latest', swmr=True) as src_side_h5:

                    # Read runw block for main and side bands
                    src_main_h5[runw_path_freq_a].read_direct(
                        main_image,
                        np.s_[row_start:row_start + block_rows_data, :])
                    src_side_h5[runw_path_freq_b].read_direct(
                        side_image,
                        np.s_[row_start:row_start + block_rows_data, :])
                    # Read coherence block for main and side bands
                    src_main_h5[rcoh_path_freq_a].read_direct(
                        main_coh_image,
                        np.s_[row_start:row_start + block_rows_data, :])
                    src_side_h5[rcoh_path_freq_b].read_direct(
                        side_coh_image,
                        np.s_[row_start:row_start + block_rows_data, :])

                    if "connected_components" in mask_type:
                        # Read connected components block for main and side
                        # bands
                        src_main_h5[rcom_path_freq_a].read_direct(
                            main_conn_image,
                            np.s_[row_start:row_start + block_rows_data, :])
                        src_side_h5[rcom_path_freq_b].read_direct(
                            side_conn_image,
                            np.s_[row_start:row_start + block_rows_data, :])

                    if "subswath_mask" in mask_type:
                        src_main_h5[subswath_mask_freq_a_path].read_direct(
                            subswath_mask_main_image,
                            np.s_[row_start:row_start + block_rows_data, :])
                        # Subswath mask may not be available when frequency B
                        # was not requested in the runconfig for interferogram
                        # In this case, we decimate subswath_mask in frequencyA
                        if subswath_mask_freq_b_path in src_side_h5:
                            src_side_h5[subswath_mask_freq_b_path].read_direct(
                                subswath_mask_side_image,
                                np.s_[row_start:row_start + block_rows_data, :])
                        else:
                            subswath_mask_side_image = decimate_freq_a_array(
                                    main_slant,
                                    side_slant,
                                    subswath_mask_main_image)

                    if iono_method == 'main_diff_ms_band':

                        with HDF5OptimizedReader(
                                name=runw_diff_str, mode='r',
                                libver='latest', swmr=True) as src_diff_h5:
                            src_diff_h5[runw_path_freq_b].read_direct(
                                diff_ms_image,
                                np.s_[
                                    row_start:row_start + block_rows_data,
                                    :]
                                    )
                            src_diff_h5[rcoh_path_freq_b].read_direct(
                                diff_ms_coh_image,
                                np.s_[
                                    row_start:row_start + block_rows_data,
                                    :]
                                    )
                            if "connected_components" in mask_type:
                                src_side_h5[rcom_path_freq_b].read_direct(
                                    diff_ms_conn_image,
                                    np.s_[row_start:row_start + block_rows_data, :])

                if bridge_algorithm_bool:
                    main_image = bridge_unwrapped_phase(
                        main_image,
                        radius=bridge_radius,
                        min_num_pixel=bridge_minimum_samples,
                        erosion_size=bridge_erosion_size,
                        ramp_type=bridge_deramp_type,
                        deramp_max_num_sample=bridge_ramp_maximum_pixel)
                    side_image = bridge_unwrapped_phase(
                        side_image,
                        radius=bridge_radius,
                        min_num_pixel=bridge_minimum_samples,
                        erosion_size=bridge_erosion_size,
                        ramp_type=bridge_deramp_type,
                        deramp_max_num_sample=bridge_ramp_maximum_pixel)

                if unwrap_correction_bool:
                    main_image = unwrapping_correction_with_filter(
                        main_image,
                        kernel_width=kernel_range_size,
                        kernel_length=kernel_azimuth_size,
                        sig_kernel_x=kernel_sigma_range,
                        sig_kernel_y=kernel_sigma_azimuth,
                        iterations=filter_iterations,
                        filter_method='convolution')
                    side_image = unwrapping_correction_with_filter(
                        side_image,
                        kernel_width=kernel_range_size,
                        kernel_length=kernel_azimuth_size,
                        sig_kernel_x=kernel_sigma_range,
                        sig_kernel_y=kernel_sigma_azimuth,
                        iterations=filter_iterations,
                        filter_method='convolution')

            # Estimate dispersive and non-dispersive phase
            dispersive, non_dispersive = iono_phase_obj.compute_disp_nondisp(
                phi_sub_low=sub_low_image,
                phi_sub_high=sub_high_image,
                phi_diff_low_high=diff_subband_image,
                phi_main=main_image,
                phi_side=side_image,
                phi_diff_ms=diff_ms_image,
                slant_main=main_slant,
                slant_side=side_slant)

            # Write dispersive and non-dispersive phase into the
            # ENVI format files
            iono_method_path = pathlib.Path(iono_path, iono_method)
            iono_method_path.mkdir(parents=True, exist_ok=True)
            iono_pol_path = pathlib.Path(iono_method_path, pol_comb_str)
            iono_pol_path.mkdir(parents=True, exist_ok=True)

            out_disp_path = os.path.join(
                iono_path, iono_method, pol_comb_str, 'dispersive')
            out_nondisp_path = os.path.join(
                iono_path, iono_method, pol_comb_str, 'non_dispersive')

            write_array(
                out_disp_path,
                dispersive,
                data_type=gdal.GDT_Float32,
                block_row=row_start,
                data_shape=[rows_output, cols_output])
            write_array(
                out_nondisp_path,
                non_dispersive,
                data_type=gdal.GDT_Float32,
                block_row=row_start,
                data_shape=[rows_output, cols_output])

            # Calculating the theoretical standard deviation of the
            # estimation based on the coherence of the interferograms
            sig_phi_iono_path = os.path.join(
                iono_path, iono_method, pol_comb_str, 'dispersive.sig')
            sig_phi_nondisp_path = os.path.join(
                iono_path, iono_method, pol_comb_str, 'nondispersive.sig')

            number_looks = rg_looks * az_looks

            iono_std, nondisp_std = iono_phase_obj.estimate_iono_std(
                main_coh=main_coh_image,
                side_coh=side_coh_image,
                low_band_coh=sub_low_coh_image,
                high_band_coh=sub_high_coh_image,
                diff_low_high_coh=diff_coh_image,
                slant_main=main_slant,
                slant_side=side_slant,
                number_looks=number_looks)

            # Write sigma of dispersive phase into the
            # ENVI format files
            write_array(
                sig_phi_iono_path,
                iono_std,
                data_type=gdal.GDT_Float32,
                block_row=row_start,
                data_shape=[rows_output, cols_output])
            write_array(
                sig_phi_nondisp_path,
                nondisp_std,
                data_type=gdal.GDT_Float32,
                block_row=row_start,
                data_shape=[rows_output, cols_output])

            # If filtering is not required, then write ionosphere phase
            # at this point.
            if not filter_bool:
                iono_hdf5_path = f'{output_pol_path}/ionospherePhaseScreen'
                write_disp_block_hdf5(
                    iono_output,
                    iono_hdf5_path,
                    dispersive,
                    rows_output,
                    row_start)

                iono_sig_hdf5_path = \
                    f'{output_pol_path}/ionospherePhaseScreenUncertainty'
                write_disp_block_hdf5(
                    iono_output,
                    iono_sig_hdf5_path,
                    iono_std,
                    rows_output,
                    row_start)
                # oversample ionosphere of frequencyB to frequencyA
                # and copy them to standard RUNW product.
                if iono_method in iono_method_sideband:
                    copy_iono_datasets(
                        iono_insar_cfg,
                        input_runw=iono_output,
                        output_runw=iono_output_runw,
                        blocksize=blocksize,
                        oversample_flag=True,
                        slant_main=main_slant,
                        slant_side=side_slant)
            else:
                info_channel.log(f'{mask_type} is used for mask construction')
                mask_array = np.ones([block_rows_data, cols_output],
                                     dtype=bool)
                if "coherence" in mask_type:
                    mask_image = iono_phase_obj.get_coherence_mask_array(
                        main_array=main_coh_image,
                        side_array=side_coh_image,
                        diff_ms_array=diff_ms_coh_image,
                        low_band_array=sub_low_coh_image,
                        high_band_array=sub_high_coh_image,
                        diff_low_high_band_array=diff_coh_image,
                        slant_main=main_slant,
                        slant_side=side_slant,
                        threshold=filter_coh_thresh)
                    mask_array = mask_array & mask_image

                if "connected_components" in mask_type:
                    mask_image = iono_phase_obj.get_conn_component_mask_array(
                        main_array=main_conn_image,
                        side_array=side_conn_image,
                        diff_ms_array=diff_ms_conn_image,
                        low_band_array=sub_low_conn_image,
                        high_band_array=sub_high_conn_image,
                        slant_main=main_slant,
                        slant_side=side_slant)
                    mask_array = mask_array & mask_image

                if "median_filter" in mask_type:
                    mask_image = iono_phase_obj.get_mask_median_filter(
                        disp=dispersive,
                        looks=number_looks,
                        threshold=median_filter_threshold,
                        median_filter_size=median_filter_size,
                        )
                    mask_array = mask_array & mask_image

                if "subswath_mask" in mask_type:
                    mask_array = iono_phase_obj.get_subswath_mask_array(
                        main_array=subswath_mask_main_image,
                        side_array=subswath_mask_side_image,
                        low_band_array=subswath_mask_image,
                        high_band_array=subswath_mask_image,
                        slant_main=main_slant,
                        slant_side=side_slant)
                    mask_array = mask_array & mask_image

                if "water" in mask_type:
                    # Extract preprocessing dictionary and open arrays
                    # water_mask_file is expected to have distance from the
                    # boundary of the water bodies. The values 0-100 represent
                    # the distance from the coastline and values from 101-200
                    # represent the distance from inland water boundaries.
                    water_mask_path = \
                            cfg["dynamic_ancillary_file_group"][
                                "water_mask_file"]
                    water_distance = project_map_to_radar(
                        cfg,
                        water_mask_path,
                        'A')
                    mask_image = water_distance[
                        row_start:row_start + block_rows_data, :] == 0
                    mask_array = mask_array & mask_image

                valid_area = iono_phase_obj.get_valid_area(
                    main_array=main_image,
                    side_array=side_image,
                    low_band_array=sub_low_image,
                    diff_ms_array=diff_ms_image,
                    diff_low_high_band_array=diff_subband_image,
                    high_band_array=sub_high_image,
                    slant_main=main_slant,
                    slant_side=side_slant,
                    invalid_value=0)

                valid_area_coh = iono_phase_obj.get_valid_area(
                    main_array=main_coh_image,
                    side_array=side_coh_image,
                    diff_ms_array=diff_ms_coh_image,
                    low_band_array=sub_low_coh_image,
                    high_band_array=sub_high_coh_image,
                    diff_low_high_band_array=diff_coh_image,
                    slant_main=main_slant,
                    slant_side=side_slant,
                    invalid_value=0)

                mask_path = os.path.join(
                    iono_path, iono_method, pol_comb_str, 'mask_array')
                # Write sigma of dispersive phase into the
                # ENVI format files
                write_array(
                    mask_path,
                    mask_array & valid_area & valid_area_coh,
                    data_type=gdal.GDT_Float32,
                    block_row=row_start,
                    data_shape=[rows_output, cols_output])

        if filter_bool:
            # if unwrapping correction technique is not requested,
            # save output to hdf5 at this point
            if not unwrap_correction_bool:
                with HDF5OptimizedReader(name=iono_output, mode='a',
                                         libver='latest', swmr=True) as dst_h5:
                    iono_hdf5_path = dst_h5[
                        f'{output_pol_path}/ionospherePhaseScreen']
                    iono_sig_hdf5_path = \
                        dst_h5[
                            f'{output_pol_path}/ionospherePhaseScreenUncertainty']

                    # low pass filtering for dispersive phase
                    iono_filter_obj.low_pass_filter(
                        input_data=out_disp_path,
                        input_std_dev=sig_phi_iono_path,
                        mask_path=mask_path,
                        filtered_output=iono_hdf5_path,
                        filtered_std_dev=iono_sig_hdf5_path,
                        lines_per_block=blocksize,
                        min_cluster_pixels=min_cluster_pixels)
                # oversample ionosphere of frequencyB to frequencyA
                # and copy them to standard RUNW product.
                if iono_method in iono_method_sideband:
                    copy_iono_datasets(
                        iono_insar_cfg,
                        input_runw=iono_output,
                        output_runw=iono_output_runw,
                        blocksize=blocksize,
                        oversample_flag=True,
                        slant_main=main_slant,
                        slant_side=side_slant)
            else:
                filt_disp_path = os.path.join(
                    iono_path, iono_method, pol_comb_str, 'filt_dispersive')
                filt_disp_sig_path = os.path.join(
                    iono_path, iono_method, pol_comb_str, 'filt_dispersive.sig')
                iono_filter_obj.low_pass_filter(
                    input_data=out_disp_path,
                    input_std_dev=sig_phi_iono_path,
                    mask_path=mask_path,
                    filtered_output=filt_disp_path,
                    filtered_std_dev=filt_disp_sig_path,
                    lines_per_block=blocksize,
                    min_cluster_pixels=min_cluster_pixels)

                # low pass filtering for non-dispersive phase
                filt_nondisp_path = os.path.join(
                    iono_path, iono_method, pol_comb_str,
                    'filt_nondispersive')
                filt_nondisp_sig_path = os.path.join(
                    iono_path, iono_method, pol_comb_str,
                    'filt_nondispersive.sig')
                iono_filter_obj.low_pass_filter(
                    input_data=out_nondisp_path,
                    input_std_dev=sig_phi_nondisp_path,
                    mask_path=mask_path,
                    filtered_output=filt_nondisp_path,
                    filtered_std_dev=filt_nondisp_sig_path,
                    lines_per_block=blocksize,
                    min_cluster_pixels=min_cluster_pixels)

                disp_tif = gdal.Open(filt_disp_path)
                disp_width = disp_tif.RasterXSize
                disp_length = disp_tif.RasterYSize
                pad_length = kernel_sigma_azimuth
                half_pad_length = int(pad_length / 2)
                correction_pad_shape = [int(2 * half_pad_length), 0]
                block_params = block_param_generator(
                    blocksize, [disp_length, cols_main],
                    correction_pad_shape)

                # Generate block_params_side for the sideband data
                # (separate instance of the generator)
                if iono_method in iono_method_sideband:
                    block_params_side = block_param_generator(
                        blocksize, [disp_length, cols_side],
                        correction_pad_shape)
                else:
                    block_params_side = repeat(None)
                for block_ind, (block_parm, block_parm_side) in enumerate(
                        zip(block_params, block_params_side)):

                    block_rows_data = block_parm.read_length
                    row_start = block * blocksize

                    if iono_method in iono_method_sideband:
                        block_parm_iono = block_parm_side
                    else:
                        block_parm_iono = block_parm

                    if (row_start + blocksize > rows_main):
                        block_rows_data = rows_main - row_start
                    else:
                        block_rows_data = blocksize

                    filt_disp = read_block_array(filt_disp_path,
                                                 block_parm_iono, 0)
                    filt_nondisp = read_block_array(filt_nondisp_path,
                                                    block_parm_iono, 0)
                    mask_image = read_block_array(mask_path,
                                                  block_parm_iono, 0)

                    # initialize arrays by setting None
                    sub_low_image = None
                    sub_high_image = None
                    main_image = None
                    side_image = None

                    if iono_method in iono_method_subbands:

                        with HDF5OptimizedReader(name=sub_low_runw_str, mode='r',
                            libver='latest', swmr=True) as src_low_h5, \
                            HDF5OptimizedReader(name=sub_high_runw_str, mode='r',
                            libver='latest', swmr=True) as src_high_h5:

                            # Read runw block for sub-bands
                            sub_low_image = read_block_array(
                                src_low_h5[runw_path_freq_a],
                                block_parm, 0)
                            sub_high_image = read_block_array(
                                src_high_h5[runw_path_freq_a],
                                block_parm, 0)

                            sub_high_image[mask_image == 0] = 0
                            sub_low_image[mask_image == 0] = 0

                        if iono_method == 'main_diff_low_high_subband':
                            with HDF5OptimizedReader(name=runw_path_insar,
                                                     mode='r',
                                                     libver='latest', swmr=True
                                                     ) as src_main_h5, \
                                HDF5OptimizedReader(name=sub_diff_runw_str,
                                                    mode='r',
                                                    libver='latest',
                                                    swmr=True) as src_diff_h5:
                                main_image = read_block_array(
                                    src_main_h5[runw_path_freq_a],
                                    block_parm, 0)
                                diff_subband_image = read_block_array(
                                    src_diff_h5[runw_path_freq_a],
                                    block_parm, 0)

                                main_image[mask_image == 0] = 0
                                diff_subband_image[mask_image == 0] = 0

                        if bridge_algorithm_bool:
                            sub_low_image = bridge_unwrapped_phase(
                                sub_low_image,
                                radius=bridge_radius,
                                min_num_pixel=bridge_minimum_samples,
                                erosion_size=bridge_erosion_size,
                                ramp_type=bridge_deramp_type,
                                deramp_max_num_sample=bridge_ramp_maximum_pixel)
                            sub_high_image = bridge_unwrapped_phase(
                                sub_high_image,
                                radius=bridge_radius,
                                min_num_pixel=bridge_minimum_samples,
                                erosion_size=bridge_erosion_size,
                                ramp_type=bridge_deramp_type,
                                deramp_max_num_sample=bridge_ramp_maximum_pixel)

                            if block_ind > 0:
                                sub_low_image, _ = compute_phase_jump(
                                    previous_low_with_pad,
                                    sub_low_image,
                                    half_pad_length)
                                sub_high_image, _ = compute_phase_jump(
                                    previous_high_with_pad,
                                    sub_high_image,
                                    half_pad_length)
                            previous_low_with_pad = sub_low_image
                            previous_high_with_pad = sub_high_image

                    if iono_method in iono_method_sideband:

                        main_image = np.empty(
                            [block_rows_data, cols_main],
                            dtype=float)
                        side_image = np.empty(
                            [block_rows_data, cols_side],
                            dtype=float)

                        with HDF5OptimizedReader(name=runw_freq_a_str,
                                                 mode='r',
                                                 libver='latest',
                                                 swmr=True) as src_main_h5, \
                            HDF5OptimizedReader(name=runw_freq_b_str,
                                                mode='r',
                                                libver='latest',
                                                swmr=True) as src_side_h5:

                            # Read runw block for sub-bands
                            main_image = read_block_array(
                                src_main_h5[runw_path_freq_a],
                                block_parm, 0)
                            side_image = read_block_array(
                                src_side_h5[runw_path_freq_b],
                                block_parm_side, 0)

                        if iono_method == 'main_diff_ms_band':
                            with HDF5OptimizedReader(name=runw_diff_str,
                                                     mode='r',
                                                     libver='latest',
                                                     swmr=True) as src_diff_h5:
                                diff_ms_image = read_block_array(
                                    src_diff_h5[runw_path_freq_b],
                                    block_parm_side, 0)

                        if bridge_algorithm_bool:
                            main_image = bridge_unwrapped_phase(
                                main_image,
                                radius=bridge_radius,
                                min_num_pixel=bridge_minimum_samples,
                                erosion_size=bridge_erosion_size,
                                ramp_type=bridge_deramp_type,
                                deramp_max_num_sample=bridge_ramp_maximum_pixel)
                            side_image = bridge_unwrapped_phase(
                                side_image,
                                radius=bridge_radius,
                                min_num_pixel=bridge_minimum_samples,
                                erosion_size=bridge_erosion_size,
                                ramp_type=bridge_deramp_type,
                                deramp_max_num_sample=bridge_ramp_maximum_pixel)

                            if block_ind > 0:
                                main_image, _ = compute_phase_jump(
                                    previous_main_with_pad,
                                    main_image,
                                    half_pad_length)
                                side_image, _ = compute_phase_jump(
                                    previous_side_with_pad,
                                    side_image,
                                    half_pad_length)
                            previous_main_with_pad = main_image
                            previous_side_with_pad = side_image

                    # Estimating phase unwrapping errors
                    com_unw_err, diff_unw_err = \
                      iono_phase_obj.compute_unwrapp_error(
                        disp_array=filt_disp,
                        nondisp_array=filt_nondisp,
                        main_runw=main_image,
                        side_runw=side_image,
                        diff_ms_runw=diff_ms_image,
                        slant_main=main_slant,
                        slant_side=side_slant,
                        low_sub_runw=sub_low_image,
                        high_sub_runw=sub_high_image,
                        diff_low_high_runw=diff_subband_image)

                    dispersive_unwcor, non_dispersive_unwcor = \
                        iono_phase_obj.compute_disp_nondisp(
                            phi_sub_low=sub_low_image,
                            phi_sub_high=sub_high_image,
                            phi_diff_low_high=diff_subband_image,
                            phi_main=main_image,
                            phi_side=side_image,
                            phi_diff_ms=diff_ms_image,
                            slant_main=main_slant,
                            slant_side=side_slant,
                            comm_unwcor_coef=com_unw_err,
                            diff_unwcor_coef=diff_unw_err)

                    out_disp_cor_path = os.path.join(
                        iono_path, iono_method, pol_comb_str,
                        'dispersive_cor')

                    out_nondisp_cor_path = os.path.join(
                        iono_path, iono_method, pol_comb_str,
                        'non_dispersive_cor')

                    write_array(
                        out_disp_cor_path,
                        dispersive_unwcor[half_pad_length:-half_pad_length, :],
                        data_type=gdal.GDT_Float32,
                        block_row=block_parm.write_start_line,
                        data_shape=[rows_output, cols_output])

                    write_array(
                        out_nondisp_cor_path,
                        non_dispersive_unwcor[half_pad_length:-half_pad_length, :],
                        data_type=gdal.GDT_Float32,
                        block_row=block_parm.write_start_line,
                        data_shape=[rows_output, cols_output])

                with HDF5OptimizedReader(name=iono_output, mode='a',
                                         libver='latest', swmr=True) as dst_h5:
                    iono_hdf5_path = dst_h5[f'{output_pol_path}/ionospherePhaseScreen']
                    iono_sig_hdf5_path = \
                        dst_h5[f'{output_pol_path}/ionospherePhaseScreenUncertainty']

                    iono_filter_obj.low_pass_filter(
                        input_data=out_disp_cor_path,
                        input_std_dev=sig_phi_iono_path,
                        mask_path=mask_path,
                        filtered_output=iono_hdf5_path,
                        filtered_std_dev=iono_sig_hdf5_path,
                        lines_per_block=blocksize,
                        min_cluster_pixels=min_cluster_pixels)
                # oversample ionosphere of frequencyB to frequencyA
                # and copyt them to standard RUNW product.
                if iono_method in iono_method_sideband:
                    copy_iono_datasets(
                        iono_insar_cfg,
                        input_runw=iono_output,
                        output_runw=iono_output_runw,
                        blocksize=blocksize,
                        oversample_flag=True,
                        slant_main=main_slant,
                        slant_side=side_slant)

    t_all_elapsed = time.time() - t_all
    info_channel.log("successfully ran Ionosphere in "
                     f"{t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    iono_runcfg = InsarIonosphereRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(iono_runcfg.cfg)
    run(iono_runcfg.cfg, runw_hdf5=out_paths['RUNW'])
