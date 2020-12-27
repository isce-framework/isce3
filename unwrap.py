'''
Wrapper for phase unwrapping 
'''

import pathlib
import time

import h5py
import journal

import pybind_isce3 as isce3
from pybind_nisar.workflows import h5_prep
from pybind_nisar.workflows.unwrap_argparse import UnwrapArgparse
from pybind_nisar.workflows.unwrap_runconfig import UnwrapRunConfig


def run(cfg: dict, input_hdf5: str, output_hdf5: str):
    '''
    run phase unwrapping (ICU only)
    '''

    # pull parameters from dictionary
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
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

    # ICU unwrapping (CPU only)
    unwrap_obj = isce3.unwrap.ICU()
    unwrap_obj = set_unwrap_attributes(unwrap_obj, unwrap_args)
    info_channel.log("Running phase unwrapping with: ")
    info_channel = write_unwrap_attributes(unwrap_obj, info_channel)

    t_all = time.time()

    with h5py.File(output_hdf5, 'a', libver='latest', swmr=True) as dst_h5:
        for freq, pol_list in freq_pols.items():
            src_freq_group_path = f'/science/LSAR/RIFG/swaths/frequency{freq}'
            dst_freq_group_path = src_freq_group_path.replace('RIFG', 'RUNW')

            for pol in pol_list:
                src_pol_group_path = f'{src_freq_group_path}/interferogram/{pol}'
                dst_pol_group_path = f'{dst_freq_group_path}/interferogram/{pol}'

                # create igram and correlation rasters
                igram_path = f'HDF5:{crossmul_path}:/' \
                             f'{src_pol_group_path}/wrappedPhase'
                corr_path = f'HDF5:{crossmul_path}:/' \
                            f'{src_pol_group_path}/phaseSigmaCoherence'

                # Get igram and correlation
                igram_raster = isce3.io.Raster(igram_path)
                corr_raster = isce3.io.Raster(corr_path)

                # Create unwrapped interferogram and connected components
                uigram_path = f'{dst_pol_group_path}/unwrappedPhase'
                conn_comp_path = f'{dst_pol_group_path}/connectedComponents'
                uigram_dataset = dst_h5[uigram_path]
                conn_comp_dataset = dst_h5[conn_comp_path]

                uigram_raster = isce3.io.Raster(f"IH5:::ID={uigram_dataset.id.id}".encode("utf-8"),
                                                update=True)
                conn_comp_raster = isce3.io.Raster(f"IH5:::ID={conn_comp_dataset.id.id}".encode("utf-8"),
                                                   update=True)
                if 'seed' in unwrap_args:
                   unwrap_obj.unwrap(uigram_raster, conn_comp_raster, igram_raster,
                                     corr_raster, seed=unwrap_args['seed'])
                else:
                   unwrap_obj.unwrap(uigram_raster, conn_comp_raster, igram_raster,
                                     corr_raster)
                del uigram_raster
                del conn_comp_raster


    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran phase unwrapping in {t_all_elapsed:.3f} seconds")


def write_unwrap_attributes(unwrap, info):
    '''
    Write unwrap attributes to info channel
    '''
    info.log(f"Number of buffer lines: {unwrap.buffer_lines}")
    info.log(f"Number of overlap lines: {unwrap.overlap_lines}")
    info.log(f"Use phase gradient neutron: {unwrap.use_phase_grad_neut}")
    info.log(f"Use intensity neutron: {unwrap.use_intensity_neut}")
    info.log(f"Phase gradient window size: {unwrap.phase_grad_win_size}")
    info.log(f"Phase gradient threshold: {unwrap.neut_phase_grad_thr}")
    info.log(f"Neutron intensity threshold: {unwrap.neut_intensity_thr}")
    info.log(f"Maximum intesity correlation threshold: {unwrap.neut_correlation_thr}")
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

    return info


def set_unwrap_attributes(unwrap, cfg: dict):
    '''
    Assign user-defined values in cfg to
    the unwrap object
    '''

    if 'buffer_lines' in cfg:
        unwrap.buffer_lines = cfg['buffer_lines']
    if 'overlap_lines' in cfg:
        unwrap.overlap_lines = cfg['overlap_lines']
    if 'use_phase_gradient_neutron' in cfg:
        unwrap.use_phase_grad_neut = cfg['use_phase_gradient_neutron']
    if 'use_intensity_neutron' in cfg:
        unwrap.use_intensity_neut = cfg['use_intensity_neutron']
    if 'phase_gradient_window_size' in cfg:
        unwrap.phase_grad_win_size = cfg['phase_gradient_window_size']
    if 'neutron_phase_gradient_threshold' in cfg:
        unwrap.neut_phase_grad_thr = cfg['neutron_phase_gradient_threshold']
    if 'neutron_intensity_threshold' in cfg:
        unwrap.neut_intensity_thr = cfg['neutron_intensity_threshold']
    if 'max_intensity_correlation_threshold' in cfg:
        unwrap.neut_correlation_thr = cfg['max_intensity_correlation_threshold']
    if 'trees_number' in cfg:
        unwrap.trees_number = cfg['trees_number']
    if 'max_branch_length' in cfg:
        unwrap.max_branch_length = cfg['max_branch_length']
    if 'pixel_spacing_ratio' in cfg:
        unwrap.ratio_dxdy = cfg['pixel_spacing_ratio']
    if 'initial_correlation_threshold' in cfg:
        unwrap.init_corr_thr = cfg['initial_correlation_threshold']
    if 'max_correlation_threshold' in cfg:
        unwrap.max_corr_thr = cfg['max_correlation_threshold']
    if 'correlation_threshold_increments' in cfg:
        unwrap.corr_incr_thr = cfg['correlation_threshold_increments']
    if 'min_tile_area' in cfg:
        unwrap.min_cc_area = cfg['min_tile_area']
    if 'bootstrap_lines' in cfg:
        unwrap.num_bs_lines = cfg['bootstrap_lines']
    if 'min_overlap_area' in cfg:
        unwrap.min_overlap_area = cfg['min_overlap_area']
    if 'phase_variance_threshold' in cfg:
        unwrap.phase_var_thr = cfg['phase_variance_threshold']

    return unwrap


if __name__ == "__main__":
    '''
    Run phase unwrapping from command line
    '''

    # Load command line args
    unwrap_parser = UnwrapArgparse()
    args = unwrap_parser.parse()

    # Get a runconfig dictionary from command line args
    unwrap_runconfig = UnwrapRunConfig(args)

    # Prepare RUNW HDF5
    out_paths = h5_prep.run(unwrap_runconfig.cfg)

    # out_paths['RIFG'] from h5_prep does not have actual interferogram data.
    # Use RIFG path from CLI or YAML
    if args.run_config_path is None:
        rifg = args.crossmul
    else:
        rifg = unwrap_runconfig.cfg['processing']['phase_unwrap']['crossmul_path']

    # Run phase unwrapping
    run(unwrap_runconfig.cfg, rifg, out_paths['RUNW'])
