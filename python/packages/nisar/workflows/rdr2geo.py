#!/usr/bin/env python3

'''
wrapper for rdr2geo
'''

import pathlib
import time

import journal
import pybind_isce3 as isce3
from nisar.products.readers import SLC
from nisar.workflows import gpu_check, runconfig
from nisar.workflows.rdr2geo_runconfig import Rdr2geoRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg):
    '''
    run rdr2geo
    '''
    # pull parameters from cfg
    input_hdf5 = cfg['InputFileGroup']['InputFilePath']
    dem_file = cfg['DynamicAncillaryFileGroup']['DEMFile']
    scratch_path = pathlib.Path(cfg['ProductPathGroup']['ScratchPath'])
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    # get params from SLC
    slc = SLC(hdf5file=input_hdf5)
    orbit = slc.getOrbit()

    # set defaults shared by both frequencies
    dem_raster = isce3.io.Raster(dem_file)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # NISAR RSLC products are always zero doppler
    grid_doppler = isce3.core.LUT2d()

    info_channel = journal.info("rdr2geo.run")
    info_channel.log("starting geocode SLC")

    # check if gpu ok to use
    use_gpu = gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                cfg['worker']['gpu_id'])
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)

    t_all = time.time()
    for freq in freq_pols.keys():
        # get frequency specific parameters
        radargrid = slc.getRadarGrid(freq)

        # create seperate directory within scratch dir for rdr2geo run
        rdr2geo_scratch_path = scratch_path / 'rdr2geo' / f'freq{freq}'
        rdr2geo_scratch_path.mkdir(parents=True, exist_ok=True)

        # init CPU or CUDA object accordingly
        if use_gpu:
            Rdr2Geo = isce3.cuda.geometry.Rdr2Geo
        else:
            Rdr2Geo = isce3.geometry.Rdr2Geo

        rdr2geo_obj = Rdr2Geo(radargrid, orbit, ellipsoid, grid_doppler)

        # run
        rdr2geo_obj.topo(dem_raster, str(rdr2geo_scratch_path))

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran rdr2geo in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    '''
    run rdr2geo from command line
    '''
    # load command line args
    rdr2geo_parser = YamlArgparse()
    args = rdr2geo_parser.parse()
    # get a runconfig dict from command line args
    rdr2geo_runconfig = Rdr2geoRunConfig(args)
    # run rdr2geo
    run(rdr2geo_runconfig.cfg)
