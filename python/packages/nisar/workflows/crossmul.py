#!/usr/bin/env python3

'''
wrapper for crossmul
'''
import pathlib
import time

import h5py
import isce3
import journal
from osgeo import gdal

gdal.UseExceptions()

from nisar.products.readers import SLC
from nisar.workflows import prepare_insar_hdf5
from nisar.workflows.compute_stats import compute_stats_real_data
from nisar.workflows.crossmul_runconfig import CrossmulRunConfig
from nisar.workflows.helpers import (complex_raster_path_from_h5,
                                     get_cfg_freq_pols)
from nisar.products.insar.product_paths import RIFGGroupsPaths
from nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg: dict, output_hdf5: str = None, resample_type='coarse',
        dump_on_disk=False, rg_looks=None, az_looks=None):
    '''
    run crossmul
    '''
    # pull parameters from cfg
    ref_hdf5 = cfg['input_file_group']['reference_rslc_file']
    sec_hdf5 = cfg['input_file_group']['secondary_rslc_file']
    crossmul_params = cfg['processing']['crossmul']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    flatten = crossmul_params['flatten']
    lines_per_block = crossmul_params['lines_per_block']

    if rg_looks == None:
        rg_looks = crossmul_params['range_looks']
    if az_looks == None:
        az_looks = crossmul_params['azimuth_looks']

    if flatten:
        flatten_path = crossmul_params['flatten_path']

    if output_hdf5 is None:
        output_hdf5 = str(scratch_path.joinpath('crossmul/product.h5'))

    # init parameters shared by frequency A and B
    ref_slc = SLC(hdf5file=ref_hdf5)
    sec_slc = SLC(hdf5file=sec_hdf5)

    error_channel = journal.error('crossmul.run')
    info_channel = journal.info("crossmul.run")
    info_channel.log("starting crossmultipy")

    # check if gpu ok to use
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                           cfg['worker']['gpu_id'])
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)
        crossmul = isce3.cuda.signal.Crossmul()
    else:
        crossmul = isce3.signal.Crossmul()

    crossmul.range_looks = rg_looks
    crossmul.az_looks = az_looks
    crossmul.oversample_factor = crossmul_params['oversample']
    crossmul.lines_per_block = lines_per_block

    # check if user provided path to raster(s) is a file or directory
    coregistered_slc_path = pathlib.Path(
        crossmul_params['coregistered_slc_path'])
    coregistered_is_file = coregistered_slc_path.is_file()
    if not coregistered_is_file and not coregistered_slc_path.is_dir():
        err_str = f"{coregistered_slc_path} is invalid; needs to be a file or directory."
        error_channel.log(err_str)
        raise ValueError(err_str)

    t_all = time.time()
    with h5py.File(output_hdf5, 'a', libver='latest') as dst_h5:
        for freq, pol_list, offset_pol_list in get_cfg_freq_pols(cfg):
            # create output product
            crossmul_dir = scratch_path / f'crossmul/freq{freq}'
            crossmul_dir.mkdir(parents=True, exist_ok=True)
            # get 2d doppler, discard azimuth dependency, and set crossmul dopplers
            ref_dopp = isce3.core.avg_lut2d_to_lut1d(
                ref_slc.getDopplerCentroid(frequency=freq))
            sec_dopp = isce3.core.avg_lut2d_to_lut1d(
                sec_slc.getDopplerCentroid(frequency=freq))
            crossmul.set_dopplers(ref_dopp, sec_dopp)

            freq_group_path = f'{RIFGGroupsPaths().SwathsPath}/frequency{freq}'

            # prepare flattening and range filter parameters
            ref_radar_grid = ref_slc.getRadarGrid(freq)
            crossmul.range_pixel_spacing = ref_radar_grid.range_pixel_spacing
            crossmul.wavelength = ref_radar_grid.wavelength

            # enable/disable flatten accordingly
            if flatten:
                # set frequency dependent range offset raster
                flatten_raster = isce3.io.Raster(
                    f'{flatten_path}/geo2rdr/freq{freq}/range.off')

                # Calculate the starting range shift between reference and secondary in meters
                sec_radar_grid = sec_slc.getRadarGrid(freq)
                rng_shift = (sec_radar_grid.starting_range -
                             ref_radar_grid.starting_range)

                crossmul.ref_sec_offset_starting_range_shift\
                    = rng_shift
            else:
                flatten_raster = None

            for pol in pol_list:
                output_dir = crossmul_dir / f'{pol}'
                output_dir.mkdir(parents=True, exist_ok=True)
                pol_group_path = f'{freq_group_path}/interferogram/{pol}'

                if dump_on_disk:
                    igram_path = f'{output_dir}/wrapped_igram_rg{rg_looks}_az{az_looks}'
                    coh_path = f'{output_dir}/coherence_rg{rg_looks}_az{az_looks}'
                    ifg_raster = isce3.io.Raster(igram_path, ref_radar_grid.width // rg_looks,
                                                 ref_radar_grid.length // az_looks, 1, gdal.GDT_CFloat32, 'ENVI')
                    coh_raster = isce3.io.Raster(coh_path, ref_radar_grid.width // rg_looks,
                                                 ref_radar_grid.length // az_looks, 1, gdal.GDT_Float32, 'ENVI')
                else:

                    # access the HDF5 dataset for a given frequency and polarization
                    ifg_dataset_path = f'{pol_group_path}/wrappedInterferogram'
                    ifg_dataset = dst_h5[ifg_dataset_path]

                    coh_dataset_path = f'{pol_group_path}/coherenceMagnitude'
                    coh_dataset = dst_h5[coh_dataset_path]

                    # compute multilook interferogram and coherence
                    # Construct the output raster directly from HDF5 dataset
                    ifg_raster = isce3.io.Raster(
                        f"IH5:::ID={ifg_dataset.id.id}".encode("utf-8"),
                        update=True)

                    # Construct the output raster directly from HDF5 dataset
                    coh_raster = isce3.io.Raster(
                        f"IH5:::ID={coh_dataset.id.id}".encode("utf-8"),
                        update=True)

                # prepare reference input raster
                c32_output_path = str(output_dir / 'reference.slc')
                raster_path, _ = complex_raster_path_from_h5(ref_slc, freq,
                                                             pol, ref_hdf5,
                                                             lines_per_block,
                                                             c32_output_path)
                ref_slc_raster = isce3.io.Raster(raster_path)

                # prepare secondary input raster
                if coregistered_is_file:
                    c32_output_path = str(output_dir / 'secondary.slc')
                    raster_path, _ = complex_raster_path_from_h5(sec_slc, freq,
                                                                 pol, sec_hdf5,
                                                                 lines_per_block,
                                                                 c32_output_path)
                else:
                    raster_path = str(coregistered_slc_path / f'{resample_type}_resample_slc/'
                                       f'freq{freq}/{pol}/coregistered_secondary.slc')

                sec_slc_raster = isce3.io.Raster(raster_path)

                # Compute multilooked interferogram and coherence raster
                crossmul.crossmul(ref_slc_raster, sec_slc_raster, ifg_raster,
                                  coh_raster, flatten_raster)

                del ifg_raster

                # Allocate raster statistics for coherence
                if not dump_on_disk:
                    compute_stats_real_data(coh_raster, coh_dataset)
                    # iterate over offset pols since they maybe different from
                    # polarizations in runconfig
                    for pol in offset_pol_list:
                        # Allocate stats for rubbersheet offsets
                        stats_offsets(dst_h5, freq, pol)

                del coh_raster

    t_all_elapsed = time.time() - t_all
    info_channel.log(
        f"successfully ran crossmul in {t_all_elapsed:.3f} seconds")


def stats_offsets(h5_ds, freq, pol):
    """
    Allocate statistics for dense offsets
    h5_ds: h5py.File
       h5py File
    freq: str
       Frequency to process (A or B)
    pol: str
       Polarization to process (HH, HV, VH, VV)
    """

    path = f'{RIFGGroupsPaths().SwathsPath}/frequency{freq}/pixelOffsets/{pol}/'
    offset_layer = ['slantRangeOffset', 'alongTrackOffset']

    for layer in offset_layer:
        offset_path = f'{path}/{layer}'
        offset_dataset = h5_ds[offset_path]
        offset_raster = isce3.io.Raster(
            f"IH5:::ID={offset_dataset.id.id}".encode("utf-8"))
        compute_stats_real_data(offset_raster, offset_dataset)


if __name__ == "__main__":
    '''
    run crossmul from command line
    '''
    # load command line args
    crossmul_parser = YamlArgparse(resample_type=True)
    args = crossmul_parser.parse()
    # extract resample type
    resample_type = args.resample_type
    # get a runconfig dict from command line args
    crossmul_runconfig = CrossmulRunConfig(args, resample_type)
    # prepare RIFG HDF5
    out_paths = prepare_insar_hdf5.run(crossmul_runconfig.cfg)
    # run crossmul
    run(crossmul_runconfig.cfg, out_paths['RIFG'], resample_type)
