#!/usr/bin/env python3
import copy
import journal
import os
import pathlib
import time

import numpy as np
from osgeo import gdal

import isce3
from isce3.atmosphere.main_band_estimation import (MainSideBandIonosphereEstimation,
                                                   MainDiffMsBandIonosphereEstimation)
from isce3.atmosphere.split_band_estimation import SplitBandIonosphereEstimation
from isce3.atmosphere.ionosphere_filter import IonosphereFilter, write_array
from isce3.io import HDF5OptimizedReader
from isce3.signal.interpolate_by_range import (decimate_freq_a_array,
                                              interpolate_freq_b_array)

from isce3.splitspectrum import splitspectrum

from nisar.products.readers import SLC
from nisar.workflows import (crossmul, dense_offsets, h5_prep,
                             filter_interferogram, prepare_insar_hdf5,
                             resample_slc_v2,
                             rubbersheet, unwrap)
from nisar.workflows.compute_stats import compute_stats_real_hdf5_dataset
from nisar.workflows.ionosphere_runconfig import InsarIonosphereRunConfig
from nisar.products.insar.product_paths import CommonPaths, RUNWGroupsPaths
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
        dst_h5[path].write_direct(data,
            dest_sel=np.s_[block_row : block_row + block_length,
                :block_width])

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

    # InSAR scratch path
    scratch_path = pathlib.Path(original_dict['scratch_path'])
    # ionosphere scratch path
    iono_dir_path = pathlib.Path(iono_insar_cfg['product_path_group'][
                'scratch_path'])
    # parameters
    blocksize = iono_insar_cfg['processing']['ionosphere_phase_correction'][
                'lines_per_block']
    freq_pols = iono_insar_cfg['processing']['input_subset'][
                'list_of_frequencies']
    runw_freq_a_str = original_dict['output_runw']
    runw_freq_b_str = iono_insar_cfg['product_path_group']['sas_output_file']
    offsets_dir = original_dict['offsets_dir']

    if iono_insar_cfg['processing']['fine_resample']['enabled']:
        resample_type = 'fine'
    else:
        resample_type = 'coarse'
    decimated_offset_dir = offsets_dir

    # Instantiate a RUNW object to get path to RUNW datasets
    runw_obj = RUNWGroupsPaths()

    # Set up for decimation
    swath_path = runw_obj.SwathsPath
    dest_freq_path = f"{swath_path}/frequencyA"
    dest_freq_path_b = f"{swath_path}/frequencyB"
    rslant_path_a = f"{dest_freq_path}/interferogram/slantRange"
    rslant_path_b = f"{dest_freq_path_b}/interferogram/slantRange"
    rspc_path_a = f"{dest_freq_path}/interferogram/slantRangeSpacing"
    rspc_path_b = f"{dest_freq_path_b}/interferogram/slantRangeSpacing"

    # Read slant range array from main and side bands
    with HDF5OptimizedReader(name=runw_freq_a_str, mode='r',
        libver='latest', swmr=True) as src_main_h5, \
        HDF5OptimizedReader(name=runw_freq_b_str, mode='r',
        libver='latest', swmr=True) as src_side_h5:

        # Read slant range block from HDF5
        main_slant = np.array(src_main_h5[rslant_path_a])
        side_slant = np.array(src_side_h5[rslant_path_b])
        spacing_main = np.array(src_main_h5[rspc_path_a])
        spacing_side = np.array(src_side_h5[rspc_path_b])

    resampling_scale_factor = float(int(np.round(spacing_side / spacing_main)))
    if resample_type == 'coarse':
        decimate_list = ['coarse']
    elif resample_type == 'fine':
        decimate_list = ['coarse', 'fine']

    for decimate_proc in decimate_list:
        if decimate_proc == 'coarse':
            coarse_offset_path = f'/geo2rdr/freqA'
            coarse_offset_b_path = f'/geo2rdr/freqB'

            offsets_path = f'{offsets_dir}/{coarse_offset_path}'
            offsets_b_path = f'{decimated_offset_dir}/{coarse_offset_b_path}'
        else:
            # We checked the existence of HH/VV offsets in resample_slc_runconfig.py
            # Select the first offsets available between HH and VV
            fine_offset_path = f'rubbersheet_offsets/freqA'
            fine_offset_b_path = f'rubbersheet_offsets/freqB'

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
                                if 'range' in b_off_path else 1 ]

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

    with HDF5OptimizedReader(name=input_runw, mode='a', libver='latest', swmr=True) as src_h5, \
        HDF5OptimizedReader(name=output_runw, mode='a', libver='latest', swmr=True) as dst_h5:

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
                        CommonPaths().RootPath, freq, pol)
                if 'interferogram' not in dst_h5[freq_path]:
                    dst_h5[freq_path].create_group('interferogram')

                if pol not in dst_h5[ifg_path]:
                    dst_h5[ifg_path].create_group(pol)

                if ('ionospherePhaseScreen' in src_h5[src_pol_path]) and \
                    ('ionospherePhaseScreen' not in dst_h5[dest_pol_path]):
                    iono_shape = src_h5[iono_path].shape
                    grids_val = 'None'
                    descr = "Split spectrum ionosphere phase screen"
                    h5_prep._create_datasets(dst_h5[dest_pol_path],
                                    iono_shape, np.float32,
                                    'ionospherePhaseScreen',
                                    descr=descr, units="radians",
                                    grids=grids_val,
                                    long_name='ionosphere phase screen')
                    descr = "Uncertainty of split spectrum ionosphere phase screen"
                    h5_prep._create_datasets(dst_h5[dest_pol_path],
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

                    for src_iono_path, dst_iono_path in zip(src_iono_paths, dst_iono_paths):
                        iono = np.empty([block_rows_data, cols_main], dtype=float)
                        src_h5[src_iono_path].read_direct(
                            iono,
                            np.s_[row_start : row_start + block_rows_data, :])
                        if oversample_flag:
                            iono = interpolate_freq_b_array(slant_main,
                                                            slant_side,
                                                            iono)
                        dst_h5[dst_iono_path].write_direct(iono,
                            dest_sel=np.s_[
                                    row_start:row_start+block_rows_data, :])

                # Add statistics to ionosphere datasets in RUNW
                for dst_iono_path in dst_iono_paths:
                    compute_stats_real_hdf5_dataset(dst_h5[dst_iono_path])

def insar_ionosphere_pair(original_cfg, runw_hdf5):
    """Run insar workflow for additional interferogram to be used for ionosphere
    estimation

    - For split_main_band, upper and lower sub-bands interferograms are created.
    - For main_side_band and main_diff_ms_band, frequency B interferograms are
      created.

    If interferograms to be used for ionosphere estimation do not exist,
    they are generated by modifying original_cfg.
    For example, the frequency A and B interferograms are requested in VV polarization
    while ionosphere is estimated in HH polarization, the addtional HH interferogram
    is created.

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


    iono_insar_cfg = original_cfg.copy()
    iono_insar_cfg['primary_executable'][
                'product_type'] = 'RUNW'

    # update processing parameter
    # water mask for ionosphere is not supported now.
    prep_wrapped_phase_cfg =  iono_insar_cfg['processing'][
        'phase_unwrap']['preprocess_wrapped_phase']
    unwrap_mask_type = prep_wrapped_phase_cfg['mask']['mask_type']

    if unwrap_mask_type == 'water':
        # Either set to a default value or delete the key entirely.
        prep_wrapped_phase_cfg['enabled'] = False

    if iono_method == 'split_main_band':
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

            run_insar_workflow(iono_insar_cfg,
                               partial_orig_cfg_dict,
                               out_paths)

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
                rerun_insar_pairs =+ 1
            else:
                del iono_insar_cfg['processing']['input_subset'][
                    'list_of_frequencies'][freq]

        if rerun_insar_pairs > 0 :
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

            run_insar_workflow(iono_insar_cfg,
                               partial_orig_cfg_dict,
                               out_paths)

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


def run_insar_workflow(iono_insar_cfg, original_dict, out_paths):
    '''Run InSAR workflow for ionosphere estimation pair without
    rdr2geo and geo2rdr steps

    - For split_main_band, rdr2geo and geo2rdr computed from insar workflow
      are used.
    - For methods using side-band, offsets are decimated and used for frequency B
      interferogram generation.

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
    '''

    # run insar for ionosphere pairs
    prepare_insar_hdf5.run(iono_insar_cfg)

    iono_freq_pol =  iono_insar_cfg['processing']['input_subset'][
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

    if 'RUNW' in out_paths:
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
    unwrap_correction_bool = filter_cfg['unwrap_correction']
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
    iono_insar_cfg = cfg.copy()

    # Run InSAR for sub-band SLCs (split-main-bands) or
    # for main and side bands for iono_freq_pols (main-side-bands)
    insar_ionosphere_pair(iono_insar_cfg, runw_hdf5)

    t_all = time.time()
    # Define methods to use subband or sideband
    iono_method_subbands = ['split_main_band']
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
    ref_meta_data_a = splitspectrum.bandpass_meta_data.load_from_slc(
        slc_product=base_ref_slc,
        freq='A')
    f0 = ref_meta_data_a.center_freq

    if iono_method in iono_method_subbands:
        # pull center frequencies from sub-bands
        high_ref_slc_str = os.path.join(split_slc_path, f"ref_high_band_slc.h5")
        low_ref_slc_str = os.path.join(split_slc_path, f"ref_low_band_slc.h5")
        high_ref_slc = SLC(hdf5file=high_ref_slc_str)
        low_ref_slc = SLC(hdf5file=low_ref_slc_str)

        high_sub_meta_data = splitspectrum.bandpass_meta_data.load_from_slc(
            slc_product=high_ref_slc,
            freq='A')
        low_sub_meta_data = splitspectrum.bandpass_meta_data.load_from_slc(
            slc_product=low_ref_slc,
            freq='A')
        f0_low = low_sub_meta_data.center_freq
        f0_high = high_sub_meta_data.center_freq

        f1 = None

        IonosphereEstimationMethod = SplitBandIonosphereEstimation

    if iono_method in iono_method_sideband:
        # pull center frequency from frequency B
        ref_meta_data_b = splitspectrum.bandpass_meta_data.load_from_slc(
            slc_product=base_ref_slc,
            freq='B')
        f1 = ref_meta_data_b.center_freq

        # find polarizations which are not processed in InSAR workflow
        if 'A' in orig_freq_pols:
            residual_pol_a =  list(set(
                iono_freq_pols['A']) - set(orig_freq_pols['A']))
        else:
            residual_pol_a = list(iono_freq_pols['A'])

        if 'B' in orig_freq_pols:
            residual_pol_b =  list(set(
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

        if iono_method in iono_method_subbands:
            # set paths for high and low sub-bands
            sub_low_runw_str = os.path.join(iono_path, 'low', 'RUNW.h5')
            sub_high_runw_str = os.path.join(iono_path, 'high', 'RUNW.h5')

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
                    iono_method, 'RUNW.h5')
            else:
                runw_freq_b_str = runw_path_insar
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

            with HDF5OptimizedReader(name=runw_freq_a_str, mode='r',
                libver='latest', swmr=True) as src_main_h5, \
                HDF5OptimizedReader(name=runw_freq_b_str, mode='r',
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

            sub_low_coh_image = None
            sub_high_coh_image = None
            main_coh_image = None
            side_coh_image = None

            sub_low_conn_image = None
            sub_high_conn_image = None
            main_conn_image = None
            side_conn_image = None

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

                if mask_type == "connected_components":
                    sub_low_conn_image = np.empty(
                        [block_rows_data, cols_main],
                        dtype=float)
                    sub_high_conn_image = np.empty(
                        [block_rows_data, cols_main],
                        dtype=float)

                with HDF5OptimizedReader(name=sub_low_runw_str, mode='r',
                    libver='latest', swmr=True) as src_low_h5, \
                    HDF5OptimizedReader(name=sub_high_runw_str, mode='r',
                    libver='latest', swmr=True) as src_high_h5:

                    # Read runw block for sub-bands
                    src_low_h5[runw_path_freq_a].read_direct(
                        sub_low_image,
                        np.s_[row_start : row_start + block_rows_data, :])
                    src_high_h5[runw_path_freq_a].read_direct(
                        sub_high_image,
                        np.s_[row_start : row_start + block_rows_data, :])
                    # Read coherence block for sub-bands
                    src_low_h5[rcoh_path_freq_a].read_direct(
                        sub_low_coh_image,
                        np.s_[row_start : row_start + block_rows_data, :])
                    src_high_h5[rcoh_path_freq_a].read_direct(
                        sub_high_coh_image,
                        np.s_[row_start : row_start + block_rows_data, :])

                    if mask_type == "connected_components":
                        # Read connected_components block for sub-bands
                        src_low_h5[rcom_path_freq_a].read_direct(
                            sub_low_conn_image,
                            np.s_[row_start : row_start + block_rows_data, :])
                        src_high_h5[rcom_path_freq_a].read_direct(
                            sub_high_conn_image,
                            np.s_[row_start : row_start + block_rows_data, :])

            if iono_method in iono_method_sideband:

                main_image = np.empty([block_rows_data, cols_main],
                    dtype=float)
                side_image = np.empty([block_rows_data, cols_side],
                    dtype=float)
                main_coh_image = np.empty([block_rows_data, cols_main],
                    dtype=float)
                side_coh_image = np.empty([block_rows_data, cols_side],
                    dtype=float)

                if mask_type == "connected_components":
                    main_conn_image = np.empty([block_rows_data, cols_main],
                        dtype=float)
                    side_conn_image = np.empty([block_rows_data, cols_side],
                        dtype=float)

                with HDF5OptimizedReader(name=runw_freq_a_str, mode='r',
                    libver='latest', swmr=True) as src_main_h5, \
                    HDF5OptimizedReader(name=runw_freq_b_str, mode='r',
                    libver='latest', swmr=True) as src_side_h5:

                    # Read runw block for main and side bands
                    src_main_h5[runw_path_freq_a].read_direct(
                        main_image,
                        np.s_[row_start : row_start + block_rows_data, :])
                    src_side_h5[runw_path_freq_b].read_direct(
                        side_image,
                        np.s_[row_start : row_start + block_rows_data, :])
                    # Read coherence block for main and side bands
                    src_main_h5[rcoh_path_freq_a].read_direct(
                        main_coh_image,
                        np.s_[row_start : row_start + block_rows_data, :])
                    src_side_h5[rcoh_path_freq_b].read_direct(
                        side_coh_image,
                        np.s_[row_start : row_start + block_rows_data, :])

                    if mask_type == "connected_components":
                        # Read connected components block for main and side bands
                        src_main_h5[rcom_path_freq_a].read_direct(
                            main_conn_image,
                            np.s_[row_start : row_start + block_rows_data, :])
                        src_side_h5[rcom_path_freq_b].read_direct(
                            side_conn_image,
                            np.s_[row_start : row_start + block_rows_data, :])

            # Estimate dispersive and non-dispersive phase
            dispersive, non_dispersive = iono_phase_obj.compute_disp_nondisp(
                phi_sub_low=sub_low_image,
                phi_sub_high=sub_high_image,
                phi_main=main_image,
                phi_side=side_image,
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

            write_array(out_disp_path,
                dispersive,
                data_type=gdal.GDT_Float32,
                block_row=row_start,
                data_shape=[rows_output, cols_output])
            write_array(out_nondisp_path,
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
                slant_main=main_slant,
                slant_side=side_slant,
                number_looks=number_looks)

            # Write sigma of dispersive phase into the
            # ENVI format files
            write_array(sig_phi_iono_path,
                iono_std,
                data_type=gdal.GDT_Float32,
                block_row=row_start,
                data_shape=[rows_output, cols_output])
            write_array(sig_phi_nondisp_path,
                nondisp_std,
                data_type=gdal.GDT_Float32,
                block_row=row_start,
                data_shape=[rows_output, cols_output])

            # If filtering is not required, then write ionosphere phase
            # at this point.
            if not filter_bool:
                iono_hdf5_path = f'{output_pol_path}/ionospherePhaseScreen'
                write_disp_block_hdf5(iono_output,
                    iono_hdf5_path,
                    dispersive,
                    rows_output,
                    row_start)

                iono_sig_hdf5_path = \
                    f'{output_pol_path}/ionospherePhaseScreenUncertainty'
                write_disp_block_hdf5(iono_output,
                    iono_sig_hdf5_path,
                    iono_std,
                    rows_output,
                    row_start)
                # oversample ionosphere of frequencyB to frequencyA
                # and copy them to standard RUNW product.
                if iono_method in iono_method_sideband:
                    copy_iono_datasets(iono_insar_cfg,
                        input_runw=iono_output,
                        output_runw=iono_output_runw,
                        blocksize=blocksize,
                        oversample_flag=True,
                        slant_main=main_slant,
                        slant_side=side_slant)

            else:
                info_channel.log(f'{mask_type} is used for mask construction')
                if mask_type == "coherence":
                    mask_array = iono_phase_obj.get_coherence_mask_array(
                        main_array=main_coh_image,
                        side_array=side_coh_image,
                        low_band_array=sub_low_coh_image,
                        high_band_array=sub_high_coh_image,
                        slant_main=main_slant,
                        slant_side=side_slant,
                        threshold=filter_coh_thresh)

                elif mask_type == "connected_components":
                    mask_array = iono_phase_obj.get_conn_component_mask_array(
                        main_array=main_conn_image,
                        side_array=side_conn_image,
                        low_band_array=sub_low_conn_image,
                        high_band_array=sub_high_conn_image,
                        slant_main=main_slant,
                        slant_side=side_slant)

                elif mask_type == "median_filter":
                    mask_array = iono_phase_obj.get_mask_median_filter(
                        disp=dispersive,
                        looks=number_looks,
                        threshold=filter_coh_thresh,
                        median_filter_size=median_filter_size)
                mask_path = os.path.join(
                    iono_path, iono_method, pol_comb_str, 'mask_array')
                # Write sigma of dispersive phase into the
                # ENVI format files
                write_array(mask_path,
                    mask_array,
                    data_type=gdal.GDT_Float32,
                    block_row=row_start,
                    data_shape=[rows_output, cols_output])

        if filter_bool:
            # if unwrapping correction technique is not requested,
            # save output to hdf5 at this point
            if not unwrap_correction_bool:
                with HDF5OptimizedReader(name=iono_output, mode='a', libver='latest', swmr=True) as dst_h5:
                    iono_hdf5_path = dst_h5[f'{output_pol_path}/ionospherePhaseScreen']
                    iono_sig_hdf5_path = \
                        dst_h5[f'{output_pol_path}/ionospherePhaseScreenUncertainty']

                    # low pass filtering for dispersive phase
                    iono_filter_obj.low_pass_filter(
                        input_data=out_disp_path,
                        input_std_dev=sig_phi_iono_path,
                        mask_path=mask_path,
                        filtered_output=iono_hdf5_path,
                        filtered_std_dev=iono_sig_hdf5_path,
                        lines_per_block=blocksize)
                # oversample ionosphere of frequencyB to frequencyA
                # and copy them to standard RUNW product.
                if iono_method in iono_method_sideband:
                    copy_iono_datasets(iono_insar_cfg,
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
                    lines_per_block=blocksize)

                # low pass filtering for non-dispersive phase
                filt_nondisp_path = os.path.join(
                    iono_path, iono_method, pol_comb_str, 'filt_nondispersive')
                filt_nondisp_sig_path = os.path.join(
                    iono_path, iono_method, pol_comb_str, 'filt_nondispersive.sig')
                iono_filter_obj.low_pass_filter(
                    input_data=out_nondisp_path,
                    input_std_dev=sig_phi_nondisp_path,
                    mask_path=mask_path,
                    filtered_output=filt_nondisp_path,
                    filtered_std_dev=filt_nondisp_sig_path,
                    lines_per_block=blocksize)

                disp_tif = gdal.Open(filt_disp_path)
                nondisp_tif = gdal.Open(filt_nondisp_path)
                disp_width = disp_tif.RasterXSize
                disp_length = disp_tif.RasterYSize

                for block in range(0, nblocks):
                    row_start = block * blocksize
                    if (row_start + blocksize > rows_main):
                        block_rows_data = rows_main - row_start
                    else:
                        block_rows_data = blocksize

                    filt_disp = disp_tif.GetRasterBand(1).ReadAsArray(0,
                        row_start,
                        disp_width,
                        block_rows_data)

                    filt_nondisp = nondisp_tif.GetRasterBand(1).ReadAsArray(0,
                        row_start,
                        disp_width,
                        block_rows_data)

                    # initialize arrays by setting None
                    sub_low_image = None
                    sub_high_image = None
                    main_image = None
                    side_image = None

                    if iono_method in iono_method_subbands:
                        sub_low_image = np.empty([block_rows_data, cols_main],
                            dtype=float)
                        sub_high_image = np.empty([block_rows_data, cols_main],
                            dtype=float)

                        with HDF5OptimizedReader(name=sub_low_runw_str, mode='r',
                            libver='latest', swmr=True) as src_low_h5, \
                            HDF5OptimizedReader(name=sub_high_runw_str, mode='r',
                            libver='latest', swmr=True) as src_high_h5:

                            # Read runw block for sub-bands
                            src_low_h5[runw_path_freq_a].read_direct(
                                sub_low_image,
                                np.s_[row_start : row_start + block_rows_data, :])
                            src_high_h5[runw_path_freq_a].read_direct(
                                sub_high_image,
                                np.s_[row_start : row_start + block_rows_data, :])

                    if iono_method in iono_method_sideband:

                        main_image = np.empty([block_rows_data, cols_main],
                            dtype=float)
                        side_image = np.empty([block_rows_data, cols_side],
                            dtype=float)

                        with HDF5OptimizedReader(name=runw_freq_a_str, mode='r',
                            libver='latest', swmr=True) as src_main_h5, \
                            HDF5OptimizedReader(name=runw_freq_b_str, mode='r',
                            libver='latest', swmr=True) as src_side_h5:

                            # Read runw block for main and side bands
                            src_main_h5[runw_path_freq_a].read_direct(
                                main_image,
                                np.s_[row_start : row_start + block_rows_data, :])
                            src_side_h5[runw_path_freq_b].read_direct(
                                side_image,
                                np.s_[row_start : row_start + block_rows_data, :])

                    # Estimating phase unwrapping errors
                    com_unw_err, diff_unw_err = iono_phase_obj.compute_unwrapp_error(
                        disp_array=filt_disp,
                        nondisp_array=filt_nondisp,
                        main_runw=main_image,
                        side_runw=side_image,
                        slant_main=main_slant,
                        slant_side=side_slant,
                        low_sub_runw=sub_low_image,
                        high_sub_runw=sub_high_image)

                    dispersive_unwcor, non_dispersive_unwcor = \
                        iono_phase_obj.compute_disp_nondisp(
                        phi_sub_low=sub_low_image,
                        phi_sub_high=sub_high_image,
                        phi_main=main_image,
                        phi_side=side_image,
                        slant_main=main_slant,
                        slant_side=side_slant,
                        comm_unwcor_coef=com_unw_err,
                        diff_unwcor_coef=diff_unw_err)

                    out_disp_cor_path = os.path.join(
                        iono_path, iono_method, pol_comb_str, f'dispersive_cor')
                    out_nondisp_cor_path = os.path.join(
                        iono_path, iono_method, pol_comb_str, 'non_dispersive_cor')

                    write_array(out_disp_cor_path,
                        dispersive_unwcor,
                        data_type=gdal.GDT_Float32,
                        block_row=row_start,
                        data_shape=[rows_output, cols_output])

                    write_array(out_nondisp_cor_path,
                        non_dispersive_unwcor,
                        data_type=gdal.GDT_Float32,
                        block_row=row_start,
                        data_shape=[rows_output, cols_output])

                with HDF5OptimizedReader(name=iono_output, mode='a', libver='latest', swmr=True) as dst_h5:
                    iono_hdf5_path = dst_h5[f'{output_pol_path}/ionospherePhaseScreen']
                    iono_sig_hdf5_path = \
                        dst_h5[f'{output_pol_path}/ionospherePhaseScreenUncertainty']

                    iono_filter_obj.low_pass_filter(
                        input_data=out_disp_cor_path,
                        input_std_dev=sig_phi_iono_path,
                        mask_path=mask_path,
                        filtered_output=iono_hdf5_path,
                        filtered_std_dev=iono_sig_hdf5_path,
                        lines_per_block=blocksize)
                # oversample ionosphere of frequencyB to frequencyA
                # and copyt them to standard RUNW product.
                if iono_method in iono_method_sideband:
                    copy_iono_datasets(iono_insar_cfg,
                        input_runw=iono_output,
                        output_runw=iono_output_runw,
                        blocksize=blocksize,
                        oversample_flag=True,
                        slant_main=main_slant,
                        slant_side=side_slant)


    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran Ionosphere in {t_all_elapsed:.3f} seconds")

if __name__ == "__main__":
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    iono_runcfg = InsarIonosphereRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(iono_runcfg.cfg)
    run(iono_runcfg.cfg, runw_hdf5=out_paths['RUNW'])
