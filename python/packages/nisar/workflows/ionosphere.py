#!/usr/bin/env python3
import copy
import os
import pathlib
import shutil
import time
from itertools import repeat
from pathlib import Path

import isce3
import journal
import numpy as np
import snaphu
from osgeo import gdal

from isce3.atmosphere.ionosphere_filter import (
    IonosphereFilter,
    read_block_array,
    unwrapping_correction_with_filter,
    write_array,
)
from isce3.atmosphere.main_band_estimation import (
    MainDiffMsBandIonosphereEstimation,
    MainSideBandIonosphereEstimation,
)
from isce3.atmosphere.split_band_estimation import (
    LowHighSubbandIonosphereEstimation,
    MainDiffLowHighSubbandIonosphereEstimation,
)
from isce3.core import crop_external_orbit
from isce3.core.block_param_generator import (
    block_param_generator,
    get_raster_block,
    write_raster_block,
)
from isce3.io import HDF5OptimizedReader
from isce3.signal.interpolate_by_range import (
    decimate_freq_a_array,
    interpolate_freq_b_array,
)
from isce3.splitspectrum import splitspectrum
from isce3.unwrap.bridge_phase import bridge_unwrapped_phase
from isce3.unwrap.preprocess import project_map_to_radar

from nisar.products.insar.product_paths import (
    CommonPaths,
    RIFGGroupsPaths,
    RUNWGroupsPaths,
)
from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.products.utils import (
    deepcopy_runconfig_and_keep_isce3_obj,
    interpret_subswath_mask,
)
from nisar.workflows import (
    crossmul,
    filter_interferogram,
    h5_prep,
    prepare_insar_hdf5,
    resample_slc_v2,
    unwrap,
)
from nisar.workflows.compute_stats import (
    compute_stats_real_data,
    compute_stats_real_hdf5_dataset,
)
from nisar.workflows.ionosphere_runconfig import InsarIonosphereRunConfig
from nisar.workflows.unwrap import (
    get_effective_looks,
    open_raster,
)
from nisar.workflows.yaml_argparse import YamlArgparse


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


def add_binary_mask_to_bit24(existing_mask,
                             new_binary_mask,
                             existing_range_array=None,
                             new_range_array=None):
    """
    Encode a boolean mask into bit 24 of an existing 32-bit integer mask.
    If shapes differ, resample new_binary_mask to existing_mask grid.
    """
    existing_mask = np.asarray(existing_mask, dtype=np.uint32)
    new_binary_mask = np.asarray(new_binary_mask, dtype=bool)

    if existing_mask.shape != new_binary_mask.shape:
        if existing_range_array is None or new_range_array is None:
            raise ValueError(
                "existing_range_array and new_range_array are required "
                "when shapes differ."
            )
        new_binary_mask = interpolate_freq_b_array(
            existing_range_array,
            new_range_array,
            new_binary_mask
        ).astype(bool)

    if existing_mask.shape != new_binary_mask.shape:
        raise ValueError(
            f"Shape mismatch after interpolation: "
            f"existing_mask={existing_mask.shape}, "
            f"new_binary_mask={new_binary_mask.shape}"
        )

    cleared_mask = existing_mask & np.uint32(0xFEFFFFFF)
    shifted_new_bits = new_binary_mask.astype(np.uint32) << 24

    return cleared_mask | shifted_new_bits


def update_hdf5_mask_bit24_block(
        h5_file,
        mask_dataset_path,
        new_binary_mask,
        row_start,
        existing_range_array=None,
        new_range_array=None):

    dst_mask = h5_file[mask_dataset_path]

    block_rows = new_binary_mask.shape[0]

    existing_mask = dst_mask[
        row_start:row_start + block_rows, :
    ]

    updated_mask = add_binary_mask_to_bit24(
        existing_mask=existing_mask,
        new_binary_mask=new_binary_mask,
        existing_range_array=existing_range_array,
        new_range_array=new_range_array)

    dst_mask[row_start:row_start + block_rows, :] = updated_mask


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


def compute_differential_phase(
        phase_first,
        phase_second,
        output_path,
        first_data_path,
        second_data_path,
        output_data_path,
        lines_per_block,
        first_slant_path=None,
        second_slant_path=None,
        subswath_mask_enabled=False,
        first_mask_path=None,
        second_mask_path=None,
        invalid_fill_value=0,
        freqB_resample_method='oversample',
        ):
    """
    Compute differential phase.

    If output_path is HDF5, the result is written to output_data_path
    inside output_path.

    If output_path is not HDF5, the result is written to one single-band
    GDAL/ENVI raster. In that case, output_data_path is ignored, and
    first_data_path / second_data_path must each contain only one dataset.
    """
    phase_first = str(phase_first)
    phase_second = str(phase_second)
    output_path = str(output_path)

    def _to_complex_if_needed(arr):
        """
        Ensure an array is complex.

        If it is already complex, return as-is.
        Otherwise, interpret values as phase in radians and convert to
        complex phase exp(j * phase).
        """
        if np.iscomplexobj(arr):
            return arr

        return np.exp(1j * arr)

    def _normalize_path_list(path, n, name):
        """Convert a scalar path or list of paths to a list."""
        if path is None:
            return None

        if isinstance(path, str):
            return [path] * n

        if len(path) != n:
            raise ValueError(
                f"`{name}` must be a string or a list with length {n}."
            )

        return path

    def _is_hdf5_file(filename):
        """Return True if filename looks like an HDF5 file."""

        suffix = Path(filename).suffix.lower()
        return suffix in [".h5", ".hdf5", ".he5"]

    if freqB_resample_method not in ["oversample", "decimate"]:
        raise ValueError(
            "`freqB_resample_method` must be either "
            "'oversample' or 'decimate'."
        )

    n_outputs = len(first_data_path)

    if len(second_data_path) != n_outputs:
        raise ValueError(
            "`first_data_path` and `second_data_path` must have the same length."
        )

    output_is_hdf5 = _is_hdf5_file(output_path)

    if output_is_hdf5:
        if output_data_path is None:
            raise ValueError(
                "`output_data_path` is required when `output_path` is HDF5."
            )

        if len(output_data_path) != n_outputs:
            raise ValueError(
                "`output_data_path` must have the same length as "
                "`first_data_path` and `second_data_path`."
            )

    else:
        # Non-HDF5 output means one single-band GDAL/ENVI raster.
        if n_outputs != 1:
            raise ValueError(
                "For GDAL/ENVI output, only one single-band raster is supported. "
                "Use one first_data_path and one second_data_path, and call "
                "compute_differential_phase() once per output raster."
            )

        # Make output_data_path iterable for the zip() loop below.
        # It is ignored for GDAL/ENVI output.
        output_data_path = [None]

    if subswath_mask_enabled:
        if first_mask_path is None or second_mask_path is None:
            raise ValueError(
                "When `subswath_mask_enabled=True`, both `first_mask_path` and "
                "`second_mask_path` must be provided."
            )

        first_mask_path = _normalize_path_list(
            first_mask_path,
            n_outputs,
            "first_mask_path"
        )

        second_mask_path = _normalize_path_list(
            second_mask_path,
            n_outputs,
            "second_mask_path"
        )

    # Only relevant for HDF5 output.
    is_same_file_first_output = (
        output_is_hdf5 and
        Path(phase_first).resolve() == Path(output_path).resolve()
    )

    is_same_file_first_second = (
        Path(phase_first).resolve() == Path(phase_second).resolve()
    )

    src_sec_h5 = None
    src_out_h5 = None

    with HDF5OptimizedReader(
            name=phase_first,
            mode='r' if not is_same_file_first_output else 'a',
            libver='latest',
            swmr=True) as src_first_h5:

        src_sec_h5 = (
            src_first_h5 if is_same_file_first_second else
            HDF5OptimizedReader(
                name=phase_second,
                mode='r',
                libver='latest',
                swmr=True
            )
        )

        if output_is_hdf5:
            src_out_h5 = (
                src_first_h5 if is_same_file_first_output else
                HDF5OptimizedReader(
                    name=output_path,
                    mode='a',
                    libver='latest',
                    swmr=True
                )
            )
        else:
            src_out_h5 = None

        resampling_flag = (
            first_slant_path is not None and second_slant_path is not None
        )

        if resampling_flag:
            main_slant = np.array(src_first_h5[first_slant_path])
            side_slant = np.array(src_sec_h5[second_slant_path])
        else:
            main_slant = None
            side_slant = None

        try:
            phase_first_raster0 = src_first_h5[first_data_path[0]]
            phase_second_raster0 = src_sec_h5[second_data_path[0]]

            if resampling_flag and freqB_resample_method == "decimate":
                output_length, output_width = phase_second_raster0.shape
            else:
                output_length, output_width = phase_first_raster0.shape
            output_width_main = phase_first_raster0.shape[1]
            output_width_side = phase_second_raster0.shape[1]

            output_width_chosen = output_width_main

            for out_ind, (first_ifg_path,
                          second_ifg_path,
                          out_ifg_path) in enumerate(
                    zip(first_data_path, second_data_path, output_data_path)):

                phase_first_raster = src_first_h5[first_ifg_path]
                phase_second_raster = src_sec_h5[second_ifg_path]

                if output_is_hdf5:
                    output_data_raster = src_out_h5[out_ifg_path]
                else:
                    # This is now a plain string, so your unchanged
                    # write_raster_block() can use gdal.Open(output_path, ...).
                    output_data_raster = output_path

                if subswath_mask_enabled:
                    first_mask_raster = src_first_h5[first_mask_path[out_ind]]
                    second_mask_raster = src_sec_h5[second_mask_path[out_ind]]
                else:
                    first_mask_raster = None
                    second_mask_raster = None

                block_params_main = block_param_generator(
                    lines_per_block,
                    [
                        phase_first_raster.shape[0],
                        phase_first_raster.shape[1]
                    ],
                    [0, 0]
                )

                block_params_side = block_param_generator(
                    lines_per_block,
                    [
                        phase_second_raster.shape[0],
                        phase_second_raster.shape[1]
                    ],
                    [0, 0]
                )

                for block_param_main, block_param_side in zip(
                        block_params_main,
                        block_params_side):

                    first_data_block = get_raster_block(
                        phase_first_raster,
                        block_param_main
                    )

                    if subswath_mask_enabled:
                        first_mask_block = get_raster_block(
                            first_mask_raster,
                            block_param_main
                        )
                    else:
                        first_mask_block = None

                    if resampling_flag:
                        second_data_block = get_raster_block(
                            phase_second_raster,
                            block_param_side
                        )

                        if subswath_mask_enabled:
                            second_mask_block = get_raster_block(
                                second_mask_raster,
                                block_param_side
                            )
                        else:
                            second_mask_block = None

                        if freqB_resample_method == "oversample":
                            # Output grid is first/main grid.
                            chosen_block_param = block_param_main
                            output_width_chosen = output_width_main

                        elif freqB_resample_method == "decimate":
                            # Output grid is second/side grid.
                            chosen_block_param = block_param_side
                            output_width_chosen = output_width_side

                    else:
                        second_data_block = get_raster_block(
                            phase_second_raster,
                            block_param_main
                        )

                        if subswath_mask_enabled:
                            second_mask_block = get_raster_block(
                                second_mask_raster,
                                block_param_main
                            )
                        else:
                            second_mask_block = None

                        chosen_block_param = block_param_main

                    # Convert real-valued phase to complex phase if needed.
                    first_data_block = _to_complex_if_needed(first_data_block)
                    second_data_block = _to_complex_if_needed(second_data_block)

                    if resampling_flag:
                        if freqB_resample_method == "oversample":
                            # Resample second/frequency-B data to first/main grid.

                            second_data_block = interpolate_freq_b_array(
                                main_slant,
                                side_slant,
                                second_data_block
                            )

                            if subswath_mask_enabled:
                                second_mask_block = interpolate_freq_b_array(
                                    main_slant,
                                    side_slant,
                                    second_mask_block
                                )

                        elif freqB_resample_method == "decimate":
                            # Resample first/frequency-A data to second/side grid.
                            first_data_block = decimate_freq_a_array(
                                main_slant,
                                side_slant,
                                first_data_block
                            )

                            if subswath_mask_enabled:
                                first_mask_block = decimate_freq_a_array(
                                    main_slant,
                                    side_slant,
                                    first_mask_block
                                )

                    if subswath_mask_enabled:
                        first_reference_valid, first_secondary_valid, _ = \
                            interpret_subswath_mask(first_mask_block)

                        second_reference_valid, second_secondary_valid, _ = \
                            interpret_subswath_mask(second_mask_block)

                        invalid = (
                            (~first_reference_valid) |
                            (~first_secondary_valid) |
                            (~second_reference_valid) |
                            (~second_secondary_valid)
                        )
                    else:
                        invalid = None

                    diff_phase = first_data_block * np.conj(second_data_block)

                    if invalid is not None:
                        diff_phase[invalid] = invalid_fill_value

                    write_array(
                        output_data_raster,
                        diff_phase,
                        data_type=gdal.GDT_CFloat32,
                        data_shape=[output_length, output_width_chosen],
                        block_row=chosen_block_param.write_start_line,
                        file_type='ENVI')

        finally:
            if not is_same_file_first_second and src_sec_h5 is not None:
                src_sec_h5.close()

            if output_is_hdf5 and not is_same_file_first_output:
                if src_out_h5 is not None:
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
    iono_radar_grid = iono_args['iono_radar_grid']

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
    if prep_wrapped_phase_cfg['enabled'] is True and \
       unwrap_mask_type == 'subswath_mask':
        subswath_mask_enabled = True

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
        if iono_radar_grid == 'main':
            freqB_resample_method = 'oversample'
        else:
            freqB_resample_method = 'decimate'

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
                if iono_radar_grid == 'side':
                    additional_runw = f'{new_scratch}/RUNW.h5'
                elif iono_radar_grid == 'main':

                    additional_runw = original_out_paths['RUNW']
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

            if iono_radar_grid == 'side':
                shutil.copy(phase_second, diff_phase_output)
            elif iono_radar_grid == 'main':
                diff_phase_output = None

            shutil.copy(additional_runw, out_paths['RUNW'])

            pol_list_a = iono_freq_pols['A']
            pol_list_b = iono_freq_pols['B']
            swath_path = RIFGGroupsPaths().SwathsPath
            runw_swath_path = RUNWGroupsPaths().SwathsPath

            first_slant_path = (
                f"{runw_swath_path}/frequencyA/interferogram/slantRange"
            )
            first_mask_path = (
                f"{runw_swath_path}/frequencyA/interferogram/mask"
            )

            second_slant_path = (
                f"{swath_path}/frequencyB/interferogram/slantRange"
            )
            second_mask_path = (
                f"{swath_path}/frequencyB/interferogram/mask"
            )

            if iono_radar_grid == 'side':

                first_data_path = [
                    f"{runw_swath_path}/frequencyA/interferogram/{pol_a}/unwrappedPhase"
                    for pol_a in pol_list_a
                ]

                second_data_path = [
                    f"{swath_path}/frequencyB/interferogram/{pol_b}/wrappedInterferogram"
                    for pol_b in pol_list_b
                ]

                output_data_path = second_data_path

                compute_differential_phase(
                    phase_first,
                    phase_second,
                    diff_phase_output,
                    first_data_path,
                    second_data_path,
                    output_data_path,
                    iono_args['lines_per_block'],
                    first_slant_path=first_slant_path,
                    second_slant_path=second_slant_path,
                    subswath_mask_enabled=subswath_mask_enabled,
                    first_mask_path=first_mask_path,
                    second_mask_path=second_mask_path,
                    freqB_resample_method=freqB_resample_method
                )

                unwrap.run(
                    iono_insar_cfg,
                    out_paths['RIFG'],
                    out_paths['RUNW']
                )

            elif iono_radar_grid == 'main':

                if len(pol_list_a) != len(pol_list_b):
                    raise ValueError(
                        "For main_diff_ms_band with iono_radar_grid='main', "
                        "`list_of_frequencies['A']` and `list_of_frequencies['B']` "
                        "must have the same number of polarizations. "
                        f"Got A={pol_list_a}, B={pol_list_b}."
                    )

                for pol_a, pol_b in zip(pol_list_a, pol_list_b):

                    diff_phase_output_pol = diff_dir / f"diff_main_side_{pol_a}_{pol_b}"

                    first_data_path = [
                        f"{runw_swath_path}/frequencyA/interferogram/{pol_a}/unwrappedPhase"
                    ]

                    second_data_path = [
                        f"{swath_path}/frequencyB/interferogram/{pol_b}/wrappedInterferogram"
                    ]

                    compute_differential_phase(
                        phase_first,
                        phase_second,
                        diff_phase_output_pol,
                        first_data_path,
                        second_data_path,
                        [None],
                        iono_args['lines_per_block'],
                        first_slant_path=first_slant_path,
                        second_slant_path=second_slant_path,
                        subswath_mask_enabled=subswath_mask_enabled,
                        first_mask_path=first_mask_path,
                        second_mask_path=second_mask_path,
                        freqB_resample_method=freqB_resample_method
                    )

                    coh_data_path_freq = (
                        f"{runw_swath_path}/frequencyA/interferogram/{pol_a}"
                        f"/coherenceMagnitude"
                    )

                    run_snaphu_with_gdal_igram(
                        iono_insar_cfg,
                        igram_path=str(diff_phase_output_pol),
                        coherence_hdf5=out_paths['RUNW'],
                        output_hdf5=out_paths['RUNW'],
                        freq='A',
                        pol=pol_a,
                        coherence_dataset_path=coh_data_path_freq
                    )

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


def run_snaphu_with_gdal_igram(
        cfg: dict,
        igram_path: str,
        coherence_hdf5: str,
        output_hdf5: str,
        freq: str,
        pol: str,
        coherence_dataset_path: str = None,
        unwrapped_dataset_path: str = None,
        connected_component_dataset_path: str = None):
    """
    Run SNAPHU unwrapping using a GDAL-supported wrapped interferogram
    and coherence stored in an HDF5 file.

    This function does not perform:
      - preprocessing
      - mask generation
      - bridge algorithm

    Parameters
    ----------
    cfg : dict
        Runconfig dictionary.
    igram_path : str
        GDAL-readable wrapped interferogram raster.
        This may be ENVI, GeoTIFF, VRT, etc.
    coherence_hdf5 : str
        HDF5 file containing coherenceMagnitude.
    output_hdf5 : str
        HDF5 file where unwrappedPhase and connectedComponents are written.
        This can be the same file as `coherence_hdf5`.
    freq : str
        Frequency name, e.g., "A" or "B".
    pol : str
        Polarization name, e.g., "HH", "HV", "VV", "VH".
    coherence_dataset_path : str, optional
        HDF5 dataset path for coherenceMagnitude.
        If None, this is inferred from RUNW/RIFG group paths.
    unwrapped_dataset_path : str, optional
        HDF5 dataset path for unwrappedPhase.
        If None, this is inferred from RUNW group paths.
    connected_component_dataset_path : str, optional
        HDF5 dataset path for connectedComponents.
        If None, this is inferred from RUNW group paths.

    Returns
    -------
    None
    """
    info_channel = journal.info("run_snaphu_with_gdal_igram")

    # Convert paths to strings for GDAL / HDF5 compatibility
    igram_path = str(igram_path)
    coherence_hdf5 = str(coherence_hdf5)
    output_hdf5 = str(output_hdf5)

    # Basic validation
    igram_ds = gdal.Open(igram_path, gdal.GA_ReadOnly)
    if igram_ds is None:
        raise ValueError(f"Cannot open wrapped interferogram with GDAL: {igram_path}")
    igram_ds = None

    coherence_hdf5_path = pathlib.Path(coherence_hdf5)
    output_hdf5_path = pathlib.Path(output_hdf5)

    if not coherence_hdf5_path.is_file():
        raise ValueError(f"Coherence HDF5 file does not exist: {coherence_hdf5}")

    if not output_hdf5_path.is_file():
        raise ValueError(f"Output HDF5 file does not exist: {output_hdf5}")

    # Pull config values
    scratch_path = pathlib.Path(cfg["product_path_group"]["scratch_path"])
    unwrap_args = cfg["processing"]["phase_unwrap"]
    snaphu_cfg = unwrap_args["snaphu"]

    # Only SNAPHU is supported in this simplified function
    algorithm = unwrap_args["algorithm"]
    if algorithm != "snaphu":
        raise ValueError(
            "run_snaphu_with_gdal_igram() only supports algorithm='snaphu'. "
            f"Current algorithm is: {algorithm}"
        )

    # Build default HDF5 dataset paths
    rifg_obj = RUNWGroupsPaths()

    src_freq_group_path = f"{rifg_obj.SwathsPath}/frequency{freq}"
    src_freq_bandwidth_group_path = (
        f"{rifg_obj.ProcessingInformationPath}/parameters"
        f"/reference/frequency{freq}"
    )

    dst_freq_group_path = src_freq_group_path.replace("RIFG", "RUNW")
    dst_pol_group_path = f"{dst_freq_group_path}/interferogram/{pol}"

    if coherence_dataset_path is None:
        coherence_dataset_path = (
            f"{src_freq_group_path}/interferogram/{pol}/coherenceMagnitude"
        )

    if unwrapped_dataset_path is None:
        unwrapped_dataset_path = f"{dst_pol_group_path}/unwrappedPhase"

    if connected_component_dataset_path is None:
        connected_component_dataset_path = (
            f"{dst_pol_group_path}/connectedComponents"
        )

    # Prepare scratch directory
    unwrap_scratch = scratch_path / f"unwrap_gdal/freq{freq}/{pol}"
    unwrap_scratch.mkdir(parents=True, exist_ok=True)

    # Open HDF5 files
    same_hdf5 = pathlib.Path(coherence_hdf5).resolve() == pathlib.Path(output_hdf5).resolve()

    t_start = time.time()

    with HDF5OptimizedReader(
            name=output_hdf5,
            mode="a",
            libver="latest",
            swmr=True) as dst_h5:

        if same_hdf5:
            src_h5 = dst_h5
            close_src = False

        else:
            src_h5 = HDF5OptimizedReader(
                name=coherence_hdf5,
                mode="r",
                libver="latest",
                swmr=True
            )
            close_src = True

        try:
            # Read input arrays
            igram_array = open_raster(igram_path)
            coh_array = src_h5[coherence_dataset_path][()]

            if igram_array.shape != coh_array.shape:
                raise ValueError(
                    "Wrapped interferogram and coherence shapes do not match:\n"
                    f"  igram shape     : {igram_array.shape}\n"
                    f"  coherence shape : {coh_array.shape}"
                )

            if dst_h5[unwrapped_dataset_path].shape != igram_array.shape:
                raise ValueError(
                    "Output unwrappedPhase dataset shape does not match input:\n"
                    f"  input shape              : {igram_array.shape}\n"
                    f"  unwrappedPhase shape     : "
                    f"{dst_h5[unwrapped_dataset_path].shape}"
                )

            if dst_h5[connected_component_dataset_path].shape != igram_array.shape:
                raise ValueError(
                    "Output connectedComponents dataset shape does not match input:\n"
                    f"  input shape                  : {igram_array.shape}\n"
                    f"  connectedComponents shape    : "
                    f"{dst_h5[connected_component_dataset_path].shape}"
                )

            # Determine nlooks
            if snaphu_cfg["nlooks"] is not None:
                nlooks = snaphu_cfg["nlooks"]
            else:
                # Try to compute effective looks from HDF5 metadata.
                # This requires the coherence_hdf5 to contain the same metadata
                # layout as the original RIFG.
                ref_slc_hdf5 = cfg["input_file_group"]["reference_rslc_file"]
                ref_orbit_ext = cfg["dynamic_ancillary_file_group"]["orbit_files"][
                    "reference_orbit_file"
                ]

                ref_slc = SLC(hdf5file=ref_slc_hdf5)
                ref_orbit = ref_slc.getOrbit()

                if ref_orbit_ext is not None:
                    external_orbit = load_orbit_from_xml(
                        ref_orbit_ext,
                        ref_slc.getRadarGrid(freq).ref_epoch
                    )
                    ref_orbit = crop_external_orbit(external_orbit, ref_orbit)

                rg_spacing = src_h5[
                    f"{src_freq_group_path}/interferogram/slantRangeSpacing"
                ][()]
                az_spacing = src_h5[
                    f"{src_freq_group_path}/interferogram/sceneCenterAlongTrackSpacing"
                ][()]
                rg_bw = src_h5[
                    f"{src_freq_bandwidth_group_path}/rangeBandwidth"
                ][()]
                az_bw = src_h5[
                    f"{src_freq_bandwidth_group_path}/azimuthBandwidth"
                ][()]

                nlooks = get_effective_looks(
                    ref_slc,
                    ref_orbit,
                    rg_spacing,
                    az_spacing,
                    rg_bw,
                    az_bw,
                    freq=freq
                )

            # Run SNAPHU
            snaphu.unwrap(
                igram_array,
                coh_array,
                nlooks,
                unw=dst_h5[unwrapped_dataset_path],
                conncomp=dst_h5[connected_component_dataset_path],
                cost=snaphu_cfg["cost_mode"],
                mask=None,
                init=snaphu_cfg["initialization_method"],
                min_conncomp_frac=snaphu_cfg["min_conncomp_frac"],
                phase_grad_window=snaphu_cfg["phase_grad_window"],
                ntiles=snaphu_cfg["ntiles"],
                tile_overlap=snaphu_cfg["tile_overlap"],
                nproc=snaphu_cfg["nproc"],
                tile_cost_thresh=snaphu_cfg["tile_cost_thresh"],
                min_region_size=snaphu_cfg["min_region_size"],
                single_tile_reoptimize=snaphu_cfg["single_tile_reoptimize"],
                regrow_conncomps=snaphu_cfg["regrow_conncomps"],
                scratchdir=unwrap_scratch,
                delete_scratch=True
            )

        finally:
            if close_src:
                src_h5.close()

    elapsed = time.time() - t_start
    info_channel.log("successfully ran iono unwrapping in "
                     f"{elapsed:.3f} seconds")


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
    iono_radar_grid = iono_args['iono_radar_grid']
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
    filling_guide_filter_method = filter_cfg['filling_guide_filter_method']
    filling_guide_median_size = filter_cfg['filling_guide_median_size']
    filling_outlier_threshold = filter_cfg['filling_outlier_threshold']
    filling_outlier_min_scale = filter_cfg['filling_outlier_min_scale']
    filling_outlier_mad_scale_factor = filter_cfg['filling_outlier_mad_scale_factor']    

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
        high_center_freq=f0_high,
        iono_radar_grid=iono_radar_grid)

    # Create object for ionosphere filter
    iono_filter_obj = IonosphereFilter(
        x_kernel=kernel_range_size,
        y_kernel=kernel_azimuth_size,
        sig_x=kernel_sigma_range,
        sig_y=kernel_sigma_azimuth,
        iteration=filter_iterations,
        filling_method=filling_method,
        guide_filter_method=filling_guide_filter_method,
        guide_median_size=filling_guide_median_size,
        outlier_threshold=filling_outlier_threshold,
        outlier_min_scale=filling_outlier_min_scale,
        mad_scale_factor=filling_outlier_mad_scale_factor,
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
            runw_path_freq_b = f"{dest_pol_path_b}/unwrappedPhase"
            rcoh_path_freq_b = f"{dest_pol_path_b}/coherenceMagnitude"
            rcom_path_freq_b = f"{dest_pol_path_b}/connectedComponents"
            rslant_path_b = f"{dest_freq_path_b}/interferogram/"\
                "slantRange"
            subswath_mask_freq_b_path = \
                f"{dest_freq_path_b}/interferogram/mask"

            # if iono_radar_grid is main, the output ionosphere will be estimated
            # on frequency A grid directly.
            if iono_radar_grid == 'main':
                runw_path_freq_diff = runw_path_freq_a
                rcoh_path_freq_diff = rcoh_path_freq_a
                output_pol_path = f"{dest_freq_path}/interferogram/{pol_a}"
            # if iono_radar_grid is side, the output ionosphere will be estimated
            # on frequency B grid and oversampled later.
            else:
                runw_path_freq_diff = runw_path_freq_b
                rcoh_path_freq_diff = rcoh_path_freq_b
                output_pol_path = f"{dest_freq_path_b}/interferogram/{pol_a}"

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
            if iono_radar_grid == 'side':
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

            if iono_radar_grid == 'main':
                rows_output = rows_main
                cols_output = cols_main
            else:
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
            diff_subband_conn_image = None

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
                            src_diff_h5[rcom_path_freq_a].read_direct(
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
                    if iono_radar_grid == 'main':
                        diff_ms_image = np.empty(
                            [block_rows_data, cols_main],
                            dtype=float)
                        diff_ms_coh_image = np.empty(
                            [block_rows_data, cols_main],
                            dtype=float)
                    else:
                        diff_ms_image = np.empty(
                            [block_rows_data, cols_side],
                            dtype=float)
                        diff_ms_coh_image = np.empty(
                            [block_rows_data, cols_side],
                            dtype=float)

                if "connected_components" in mask_type:
                    main_conn_image = np.empty(
                        [block_rows_data, cols_main],
                        dtype=float)
                    side_conn_image = np.empty(
                        [block_rows_data, cols_side],
                        dtype=float)
                    if iono_method == 'main_diff_ms_band':
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
                            src_diff_h5[runw_path_freq_diff].read_direct(
                                diff_ms_image,
                                np.s_[
                                    row_start:row_start + block_rows_data,
                                    :]
                                    )
                            src_diff_h5[rcoh_path_freq_diff].read_direct(
                                diff_ms_coh_image,
                                np.s_[
                                    row_start:row_start + block_rows_data,
                                    :]
                                    )
                            if "connected_components" in mask_type:
                                src_diff_h5[rcom_path_freq_b].read_direct(
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

                    if iono_method == 'main_side_band':
                        side_image = bridge_unwrapped_phase(
                            side_image,
                            radius=bridge_radius,
                            min_num_pixel=bridge_minimum_samples,
                            erosion_size=bridge_erosion_size,
                            ramp_type=bridge_deramp_type,
                            deramp_max_num_sample=bridge_ramp_maximum_pixel)
                    elif iono_method == 'main_diff_ms_band':
                        diff_ms_image = bridge_unwrapped_phase(
                            diff_ms_image,
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

                    if iono_method == 'main_side_band':
                        side_image = unwrapping_correction_with_filter(
                            side_image,
                            kernel_width=kernel_range_size,
                            kernel_length=kernel_azimuth_size,
                            sig_kernel_x=kernel_sigma_range,
                            sig_kernel_y=kernel_sigma_azimuth,
                            iterations=filter_iterations,
                            filter_method='convolution')
                    elif iono_method == 'main_diff_ms_band':
                        diff_ms_image = unwrapping_correction_with_filter(
                            diff_ms_image,
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
                diff_ms_coh=diff_ms_coh_image,
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
                if iono_method in iono_method_sideband and iono_radar_grid == 'side':
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
                        diff_low_high_band_array=diff_subband_conn_image,
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
                    mask_subswath = iono_phase_obj.get_subswath_mask_array(
                        main_array=subswath_mask_main_image,
                        side_array=subswath_mask_side_image,
                        low_band_array=subswath_mask_image,
                        high_band_array=subswath_mask_image,
                        slant_main=main_slant,
                        slant_side=side_slant)
                    mask_array &= mask_subswath

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
                final_iono_mask = mask_array & valid_area & valid_area_coh

                mask_path = os.path.join(
                    iono_path, iono_method, pol_comb_str, 'mask_array')
                # Write sigma of dispersive phase into the
                # ENVI format files
                write_array(
                    mask_path,
                    final_iono_mask,
                    data_type=gdal.GDT_Float32,
                    block_row=row_start,
                    data_shape=[rows_output, cols_output])

                if iono_method in iono_method_sideband:
                    existing_range = main_slant
                    new_range = side_slant

                else:
                    existing_range = None
                    new_range = None

                with HDF5OptimizedReader(name=runw_path_insar, mode='a',
                                         libver='latest') as dst_h5:
                    update_hdf5_mask_bit24_block(
                        dst_h5,
                        subswath_mask_freq_a_path,
                        final_iono_mask,
                        row_start=row_start,
                        existing_range_array=existing_range,
                        new_range_array=new_range)

        if filter_bool:
            # if unwrapping correction technique is not requested,
            # save output to hdf5 at this point
            if not unwrap_correction_bool:
                with HDF5OptimizedReader(name=iono_output, mode='a',
                                         libver='latest') as dst_h5:
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
                        min_cluster_pixels=min_cluster_pixels
                        )
                # oversample ionosphere of frequencyB to frequencyA
                # and copy them to standard RUNW product.
                if iono_method in iono_method_sideband and iono_radar_grid == 'side':
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
                    min_cluster_pixels=min_cluster_pixels,
                    )

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
                    min_cluster_pixels=min_cluster_pixels,
                    )

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
                    row_start = block_ind * blocksize

                    if iono_method in iono_method_sideband:
                        if iono_radar_grid == 'main':
                            block_parm_iono = block_parm
                        else:
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
                            [block_rows_data, cols_output],
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

                            if iono_method == 'main_side_band':
                                side_image = read_block_array(
                                    src_side_h5[runw_path_freq_b],
                                    block_parm_iono, 0)

                        if iono_method == 'main_diff_ms_band':
                            with HDF5OptimizedReader(name=runw_diff_str,
                                                     mode='r',
                                                     libver='latest',
                                                     swmr=True) as src_diff_h5:
                                diff_ms_image = read_block_array(
                                    src_diff_h5[runw_path_freq_diff],
                                    block_parm_iono, 0)

                        if bridge_algorithm_bool:
                            main_image = bridge_unwrapped_phase(
                                main_image,
                                radius=bridge_radius,
                                min_num_pixel=bridge_minimum_samples,
                                erosion_size=bridge_erosion_size,
                                ramp_type=bridge_deramp_type,
                                deramp_max_num_sample=bridge_ramp_maximum_pixel)

                            if iono_method == 'main_side_band':
                                side_image = bridge_unwrapped_phase(
                                    side_image,
                                    radius=bridge_radius,
                                    min_num_pixel=bridge_minimum_samples,
                                    erosion_size=bridge_erosion_size,
                                    ramp_type=bridge_deramp_type,
                                    deramp_max_num_sample=bridge_ramp_maximum_pixel)

                            elif iono_method == 'main_diff_ms_band':
                                diff_ms_image = bridge_unwrapped_phase(
                                    diff_ms_image,
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
                                if iono_method == 'main_side_band':
                                    side_image, _ = compute_phase_jump(
                                        previous_side_with_pad,
                                        side_image,
                                        half_pad_length)
                                elif iono_method == 'main_diff_ms_band':
                                    diff_ms_image, _ = compute_phase_jump(
                                        previous_diff_ms_with_pad,
                                        diff_ms_image,
                                        half_pad_length)
                            previous_main_with_pad = main_image
                            previous_side_with_pad = side_image
                            previous_diff_ms_with_pad = diff_ms_image

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
                                         libver='latest') as dst_h5:
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
                        min_cluster_pixels=min_cluster_pixels,
                        )
                # oversample ionosphere of frequencyB to frequencyA
                # and copyt them to standard RUNW product.
                if iono_method in iono_method_sideband and iono_radar_grid == 'side':
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
