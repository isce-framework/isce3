#!/usr/bin/env python3

'''
wrapper for rdr2geo
'''

import pathlib
import time

from osgeo import gdal

import journal
import isce3
from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows.rdr2geo_runconfig import Rdr2geoRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse

def get_raster_obj(out_path: str, radargrid: isce3.product.RadarGridParameters,
                   write2disk: bool, dtype: int) -> None:
    '''Function that returns io.Raster or None based on write2disk bool

    dtype has to be a GDAL datatype
    '''
    if not write2disk:
        return None

    return isce3.io.Raster(out_path, radargrid.width, radargrid.length, 1,
                           dtype, 'ENVI')

def run(cfg):
    '''
    run rdr2geo
    '''
    # pull parameters from cfg
    input_hdf5 = cfg['input_file_group']['reference_rslc_file']
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']
    ref_orbit = cfg['dynamic_ancillary_file_group']['orbit_files']['reference_orbit_file']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    threshold = cfg['processing']['rdr2geo']['threshold']
    numiter = cfg['processing']['rdr2geo']['numiter']
    extraiter = cfg['processing']['rdr2geo']['extraiter']
    lines_per_block = cfg['processing']['rdr2geo']['lines_per_block']

    # get params from SLC
    slc = SLC(hdf5file=input_hdf5)

    # Get orbit
    if ref_orbit is not None:
        # SLC will get first radar grid whose frequency is available.
        # Reference epoch and orbit have no frequency dependency.
        orbit = load_orbit_from_xml(ref_orbit, slc.getRadarGrid().ref_epoch)
    else:
        orbit = slc.getOrbit()

    # set defaults shared by both frequencies
    dem_raster = isce3.io.Raster(dem_file)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # NISAR RSLC products are always zero doppler
    grid_doppler = isce3.core.LUT2d()

    info_channel = journal.info("rdr2geo.run")
    info_channel.log("starting rdr2geo")

    # check if gpu ok to use
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
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

        rdr2geo_obj = Rdr2Geo(radargrid, orbit, ellipsoid, grid_doppler,
                              threshold=threshold, numiter=numiter,
                              extraiter=extraiter,
                              lines_per_block=lines_per_block)

        # dict of layer names keys to tuples of their output name and GDAL types
        layers = {'x':('x', gdal.GDT_Float64), 'y':('y', gdal.GDT_Float64),
                  'z':('z', gdal.GDT_Float64),
                  'incidence':('incidence', gdal.GDT_Float32),
                  'heading':('heading', gdal.GDT_Float32),
                  'local_incidence':('localIncidence', gdal.GDT_Float32),
                  'local_psi':('localPsi', gdal.GDT_Float32),
                  'simulated_amplitude':('simamp', gdal.GDT_Float32),
                  'layover_shadow':('layoverShadowMask', gdal.GDT_Byte)}

        # get rdr2geo config dict from processing dict for brevity
        rdr2geo_cfg = cfg['processing']['rdr2geo']

        # list comprehend rasters to be written from layers dict
        raster_list = [
            get_raster_obj(f'{str(rdr2geo_scratch_path)}/{fname}.rdr',
                           radargrid, rdr2geo_cfg[f'write_{key_name}'],
                           dtype)
            for key_name, (fname, dtype) in layers.items()]

        # extract individual elements from dict as args for topo
        x_raster, y_raster, height_raster, incidence_raster,\
            heading_raster, local_incidence_raster, local_psi_raster,\
            simulated_amplitude_raster, shadow_raster = raster_list

        # run topo - with east and north unit vector components of ground to
        # satellite layers permanently disabled.
        rdr2geo_obj.topo(dem_raster, x_raster, y_raster, height_raster,
                         incidence_raster, heading_raster, local_incidence_raster,
                         local_psi_raster, simulated_amplitude_raster,
                         shadow_raster, None, None)

        # remove undesired/None rasters from raster list
        raster_list = [raster for raster in raster_list if raster is not None]

        # save non-None rasters to vrt
        output_vrt = isce3.io.Raster(f'{str(rdr2geo_scratch_path)}/topo.vrt', raster_list)
        output_vrt.set_epsg(rdr2geo_obj.epsg_out)

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
