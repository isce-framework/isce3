'''
wrapper for rdr2geo
'''

import pathlib
import sys
import time

import journal

import pybind_isce3 as isce3
from pybind_nisar.products.readers import SLC
from pybind_nisar.workflows import runconfig, gpu_check
from pybind_nisar.workflows.rdr2geo_argparse import Rdr2geoArgparse
from pybind_nisar.workflows.rdr2geo_runconfig import Rdr2geoRunConfig


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
        doppler = slc.getDopplerCentroid(freq)

        # create seperate directory within scratch dir for rdr2geo run
        rdr2geo_scratch_path = scratch_path / f'rdr2geo{freq}'
        rdr2geo_scratch_path.mkdir(exist_ok=True)

        # init CPU or CUDA object accordingly
        if use_gpu:
            rdr2geo_obj = isce3.cuda.geometry.Rdr2Geo(radargrid, orbit,
                                                      ellipsoid, doppler)
        else:
            rdr2geo_obj = isce3.geometry.Rdr2Geo(radargrid, orbit, ellipsoid,
                                                 doppler)

        # run
        rdr2geo_obj.topo(dem_raster, str(rdr2geo_scratch_path))

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran rdr2geo in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    '''
    run rdr2geo from command line
    '''
    # load command line args
    rdr2geo_parser = Rdr2geoArgparse()
    args = rdr2geo_parser.parse()
    # get a runconfig dict from command line args
    rdr2geo_runconfig = Rdr2geoRunConfig(args)
    # run rdr2geo
    run(rdr2geo_runconfig.cfg)
