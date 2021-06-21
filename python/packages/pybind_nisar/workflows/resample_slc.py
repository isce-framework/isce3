#!/usr/bin/env python3

'''
wrapper for resample
'''

import pathlib
import time

import journal
import pybind_isce3 as isce3
from osgeo import gdal
from pybind_nisar.products.readers import SLC
from pybind_nisar.workflows import gpu_check, runconfig
from pybind_nisar.workflows.resample_slc_runconfig import ResampleSlcRunConfig
from pybind_nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg):
    '''
    run resample_slc
    '''
    input_hdf5 = cfg['InputFileGroup']['SecondaryFilePath']
    scratch_path = pathlib.Path(cfg['ProductPathGroup']['ScratchPath'])
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    resamp_args = cfg['processing']['resample']

    # Get SLC parameters
    slc = SLC(hdf5file=input_hdf5)

    info_channel = journal.info('resample_slc.run')
    info_channel.log('starting resampling SLC')

    # Check if use GPU or CPU resampling
    use_gpu = gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                cfg['worker']['gpu_id'])

    if use_gpu:
        # Set current CUDA device
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)

    t_all = time.time()

    for freq in freq_pols.keys():
        # Get frequency specific parameters
        radar_grid = slc.getRadarGrid(frequency=freq)
        native_doppler = slc.getDopplerCentroid(frequency=freq)

        # create separate directory within scratch for resample_slc
        resample_slc_scratch_path = scratch_path / 'resample_slc' / f'freq{freq}'
        resample_slc_scratch_path.mkdir(parents=True, exist_ok=True)

        # Initialize CPU or GPU resample object accordingly
        if use_gpu:
            Resamp = isce3.cuda.image.ResampSlc
        else:
            Resamp = isce3.image.ResampSlc

        resamp_obj = Resamp(radar_grid, native_doppler, radar_grid.wavelength)

        # If lines per tile is > 0, assign it to resamp_obj
        if resamp_args['lines_per_tile']:
            resamp_obj.lines_per_tile = resamp_args['lines_per_tile']

        # Open offsets
        offset_dir = pathlib.Path(cfg['processing']['resample']['offset_dir'])
        geo2rdr_off_path = offset_dir / 'geo2rdr' / f'freq{freq}'

        # Open offsets
        rg_off = isce3.io.Raster(str(geo2rdr_off_path / 'range.off'))
        az_off = isce3.io.Raster(str(geo2rdr_off_path / 'azimuth.off'))

        # Get polarization list to process
        pol_list = freq_pols[freq]

        for pol in pol_list:
            # Create directory for each polarization
            out_dir = resample_slc_scratch_path / pol
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / 'coregistered_secondary.slc'

            # Extract and create raster of SLC to resample
            h5_ds = f'/{slc.SwathPath}/frequency{freq}/{pol}'
            raster_path = f'HDF5:{input_hdf5}:{h5_ds}'
            raster = isce3.io.Raster(raster_path)

            # Create output raster
            resamp_slc = isce3.io.Raster(str(out_path), rg_off.width,
                                         rg_off.length, rg_off.num_bands, gdal.GDT_CFloat32, 'ENVI')
            resamp_obj.resamp(raster, resamp_slc, rg_off, az_off)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran resample in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    '''
    run resample_slc from command line
    '''

    # load command line args
    resample_slc_parser = YamlArgparse()
    args = resample_slc_parser.parse()

    # Get a runconfig dictionary from command line args
    resample_slc_runconfig = ResampleSlcRunConfig(args)

    # Run resample_slc
    run(resample_slc_runconfig.cfg)
