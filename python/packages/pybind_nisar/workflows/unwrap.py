#!/usr/bin/env python3

'''
Wrapper for phase unwrapping
'''

import pathlib
import time

import h5py
import journal
import os
import pybind_isce3 as isce3
from osgeo import gdal
import numpy as np
from pybind_nisar.workflows import h5_prep
from pybind_nisar.workflows.unwrap_runconfig import UnwrapRunConfig
from pybind_nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg: dict, input_hdf5: str, output_hdf5: str):
    '''
    run phase unwrapping (ICU only)
    '''

    # pull parameters from dictionary
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    scratch_path = pathlib.Path(cfg['ProductPathGroup']['ScratchPath'])
    unwrap_args = cfg['processing']['phase_unwrap']

    # Create error and info channels
    error_channel = journal.error('unwrap.run')
    info_channel = journal.info("unwrap.run")
    info_channel.log("Starting phase unwrapping")

    crossmul_path = pathlib.Path(input_hdf5)
    # if not file or directory raise error
    if not crossmul_path.is_file():
        err_str = f"{crossmul_path} is invalid; needs to be a file"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Instantiate correct unwrap object
    if unwrap_args['algorithm'] == 'icu':
        unwrap_obj = isce3.unwrap.ICU()
    elif unwrap_args['algorithm'] == 'phass':
        unwrap_obj = isce3.unwrap.Phass()
    else:
        err_str = "Invalid unwrapping algorithm"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Depending on unwrapper, set unwrap attributes
    unwrap_obj = set_unwrap_attributes(unwrap_obj, unwrap_args)
    info_channel.log("Running phase unwrapping with: ")
    log_unwrap_attributes(unwrap_obj, info_channel,
                          unwrap_args['algorithm'])

    t_all = time.time()

    with h5py.File(output_hdf5, 'a', libver='latest', swmr=True) as dst_h5,\
         h5py.File(crossmul_path, 'r', libver='latest', swmr=True) as src_h5:
        for freq, pol_list in freq_pols.items():
            src_freq_group_path = f'/science/LSAR/RIFG/swaths/frequency{freq}'
            dst_freq_group_path = src_freq_group_path.replace('RIFG', 'RUNW')

            for pol in pol_list:
                src_pol_group_path = f'{src_freq_group_path}/interferogram/{pol}'
                dst_pol_group_path = f'{dst_freq_group_path}/interferogram/{pol}'

                # Interferogram filepath
                igram_path = f'HDF5:{crossmul_path}:/' \
                             f'{src_pol_group_path}/wrappedInterferogram'

                # Prepare correlation input raster
                corr_path = f'HDF5:{crossmul_path}:/' \
                            f'{src_pol_group_path}/coherenceMagnitude'
                corr_raster = isce3.io.Raster(corr_path)

                # Create unwrapped interferogram output raster
                uigram_path = f'{dst_pol_group_path}/unwrappedPhase'
                uigram_dataset = dst_h5[uigram_path]
                uigram_raster = isce3.io.Raster(f"IH5:::ID={uigram_dataset.id.id}".encode("utf-8"),
                                                update=True)

                # Create connected components output raster
                conn_comp_path = f'{dst_pol_group_path}/connectedComponents'
                conn_comp_dataset = dst_h5[conn_comp_path]
                conn_comp_raster = isce3.io.Raster(f"IH5:::ID={conn_comp_dataset.id.id}".encode("utf-8"),
                                                   update=True)

                # If unwrapping algorithm is ICU, run it with or without seed
                if unwrap_args['algorithm'] == 'icu':
                    # Allocate interferogram as ISCE3 raster
                    igram_raster = isce3.io.Raster(igram_path)
                    if 'seed' in unwrap_args['icu']:
                        unwrap_obj.unwrap(uigram_raster, conn_comp_raster,
                                          igram_raster,
                                          corr_raster, seed=unwrap_args['seed'])
                    else:
                        unwrap_obj.unwrap(uigram_raster, conn_comp_raster,
                                          igram_raster,
                                          corr_raster)
                else:
                    # Unwrapping algorithm is PHASS which requires
                    # the interferometric phase as input raster
                    unwrap_scratch = scratch_path / f'unwrap/freq{freq}/{pol}'
                    unwrap_scratch.mkdir(parents=True, exist_ok=True)

                    # Using GDAL pixel function to compute a wrapped phase VRT
                    ds = gdal.Open(igram_path, gdal.GA_ReadOnly)
                    vrttmpl = f'''
                            <VRTDataset rasterXSize="{ds.RasterXSize}" rasterYSize="{ds.RasterYSize}">
                            <VRTRasterBand dataType="Float32" band="1" subClass="VRTDerivedRasterBand">
                            <Description>Phase</Description>
                            <PixelFunctionType>phase</PixelFunctionType>
                            <SimpleSource>
                            <SourceFilename>{igram_path}</SourceFilename>
                            </SimpleSource>
                            </VRTRasterBand>
                            </VRTDataset>'''
                    ds = None
                    with open(os.path.join(unwrap_scratch, 'wrapped_phase.vrt'),
                              'w') as fid:
                        fid.write(vrttmpl)

                    # Open phase_raster as ISCE3 raster
                    phase_raster = isce3.io.Raster(os.path.join(unwrap_scratch, 'wrapped_phase.vrt'))

                    # Check if power raster has been allocated
                    if 'power' in unwrap_args['phass']:
                        power_raster = isce3.io.Raster(unwrap_args['phass']['power'])
                        unwrap_obj.unwrap(phase_raster, power_raster,
                                          corr_raster, uigram_raster, conn_comp_raster)
                    else:
                        unwrap_obj.unwrap(phase_raster, corr_raster,
                                          uigram_raster, conn_comp_raster)

                if 'seed' in unwrap_args:
                    unwrap_obj.unwrap(uigram_raster, conn_comp_raster, igram_raster,
                                      corr_raster, seed=unwrap_args['seed'])
                else:
                    unwrap_obj.unwrap(uigram_raster, conn_comp_raster, igram_raster,
                                      corr_raster)

                # Copy coherence magnitude and culled offsets to RIFG
                dataset_names = ['coherenceMagnitude', 'alongTrackOffset',
                                 'slantRangeOffset']
                group_names = ['interferogram', 'pixelOffsets', 'pixelOffsets']
                for dataset_name, group_name in zip(dataset_names, group_names):
                    dst_path = f'{dst_freq_group_path}/{group_name}/{pol}/{dataset_name}'
                    src_path = f'{src_freq_group_path}/{group_name}/{pol}/{dataset_name}'
                    dst_h5[dst_path][:, :] = src_h5[src_path][()]

                del uigram_raster
                del conn_comp_raster

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran phase unwrapping in {t_all_elapsed:.3f} seconds")


def log_unwrap_attributes(unwrap, info, algorithm):
    '''
    Write unwrap attributes to info
    channel depending on unwrapping algorithm
    '''
    info.log(f"Unwrapping algorithm:{algorithm}")
    if algorithm == 'icu':
        info.log(f"Number of buffer lines: {unwrap.buffer_lines}")
        info.log(f"Number of overlap lines: {unwrap.overlap_lines}")
        info.log(f"Use phase gradient neutron: {unwrap.use_phase_grad_neut}")
        info.log(f"Use intensity neutron: {unwrap.use_intensity_neut}")
        info.log(f"Phase gradient window size: {unwrap.phase_grad_win_size}")
        info.log(f"Phase gradient threshold: {unwrap.neut_phase_grad_thr}")
        info.log(f"Neutron intensity threshold: {unwrap.neut_intensity_thr}")
        info.log(f"Maximum intensity correlation "
                 f"threshold: {unwrap.neut_correlation_thr}")
        info.log(f"Number of trees: {unwrap.trees_number}")
        info.log(f"Maximum branch length: {unwrap.max_branch_length}")
        info.log(f"Pixel spacing ratio: {unwrap.ratio_dxdy}")
        info.log(f"Initial correlation threshold: {unwrap.init_corr_thr}")
        info.log(f"Maximum correlation threshold: {unwrap.max_corr_thr}")
        info.log(f"Correlation threshold increments: {unwrap.corr_incr_thr}")
        info.log(f"Minimum tile area fraction: {unwrap.min_cc_area}")
        info.log(f"Number of bootstraping lines: {unwrap.num_bs_lines}")
        info.log(f"Minimum overlapping area: {unwrap.min_overlap_area}")
        info.log(f"Phase variance threshold: {unwrap.phase_var_thr}")
    else:
        info.log(f"Good correlation: {unwrap.good_correlation}")
        info.log(
            f"Minimum size of an unwrapped region: {unwrap.min_pixels_region}")

    return info


def set_unwrap_attributes(unwrap, cfg: dict):
    '''
    Assign user-defined values in cfg to
    the unwrap object
    '''
    error_channel = journal.error('unwrap.set_unwrap_attributes')

    algorithm = cfg['algorithm']
    if 'correlation_threshold_increments' in cfg:
        unwrap.corr_incr_thr = cfg['correlation_threshold_increments']

    if algorithm == 'icu':
        icu_cfg = cfg['icu']
        if 'buffer_lines' in icu_cfg:
            unwrap.buffer_lines = icu_cfg['buffer_lines']
        if 'overlap_lines' in icu_cfg:
            unwrap.overlap_lines = icu_cfg['overlap_lines']
        if 'use_phase_gradient_neutron' in icu_cfg:
            unwrap.use_phase_grad_neut = icu_cfg['use_phase_gradient_neutron']
        if 'use_intensity_neutron' in icu_cfg:
            unwrap.use_intensity_neut = icu_cfg['use_intensity_neutron']
        if 'phase_gradient_window_size' in icu_cfg:
            unwrap.phase_grad_win_size = icu_cfg['phase_gradient_window_size']
        if 'neutron_phase_gradient_threshold' in icu_cfg:
            unwrap.neut_phase_grad_thr = icu_cfg[
                'neutron_phase_gradient_threshold']
        if 'neutron_intensity_threshold' in icu_cfg:
            unwrap.neut_intensity_thr = icu_cfg['neutron_intensity_threshold']
        if 'max_intensity_correlation_threshold' in icu_cfg:
            unwrap.neut_correlation_thr = icu_cfg[
                'max_intensity_correlation_threshold']
        if 'trees_number' in icu_cfg:
            unwrap.trees_number = icu_cfg['trees_number']
        if 'max_branch_length' in icu_cfg:
            unwrap.max_branch_length = icu_cfg['max_branch_length']
        if 'pixel_spacing_ratio' in icu_cfg:
            unwrap.ratio_dxdy = icu_cfg['pixel_spacing_ratio']
        if 'initial_correlation_threshold' in icu_cfg:
            unwrap.init_corr_thr = icu_cfg['initial_correlation_threshold']
        if 'max_correlation_threshold' in icu_cfg:
            unwrap.max_corr_thr = icu_cfg['max_correlation_threshold']
        if 'min_tile_area' in icu_cfg:
            unwrap.min_cc_area = icu_cfg['min_tile_area']
        if 'bootstrap_lines' in icu_cfg:
            unwrap.num_bs_lines = icu_cfg['bootstrap_lines']
        if 'min_overlap_area' in icu_cfg:
            unwrap.min_overlap_area = icu_cfg['min_overlap_area']
        if 'phase_variance_threshold' in icu_cfg:
            unwrap.phase_var_thr = icu_cfg['phase_variance_threshold']
    elif algorithm == 'phass':
        phass_cfg = cfg['phass']
        if 'good_correlation' in phass_cfg:
            unwrap.good_correlation = phass_cfg['good_correlation']
        if 'min_unwrap_area' in phass_cfg:
            unwrap.min_pixels_region = phass_cfg['min_unwrap_area']
    else:
        err_str = "Not a valid unwrapping algorithm"
        error_channel.log(err_str)
        raise ValueError(err_str)

    return unwrap


if __name__ == "__main__":
    '''
    Run phase unwrapping from command line
    '''

    # Load command line args
    unwrap_parser = YamlArgparse()
    args = unwrap_parser.parse()

    # Get a runconfig dictionary from command line args
    unwrap_runconfig = UnwrapRunConfig(args)

    # Prepare RUNW HDF5
    unwrap_runconfig.cfg['PrimaryExecutable']['ProductType'] = 'RUNW_STANDALONE'
    out_paths = h5_prep.run(unwrap_runconfig.cfg)

    # Use RIFG from crossmul_path
    rifg_h5 = unwrap_runconfig.cfg['processing']['phase_unwrap']['crossmul_path']

    # Run phase unwrapping
    run(unwrap_runconfig.cfg, rifg_h5, out_paths['RUNW'])
