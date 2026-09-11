#!/usr/bin/env python3

'''
wrapper for rdr2geo
'''

import pathlib
import time

from osgeo import gdal

import journal
import isce3
from isce3.core import crop_external_orbit
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

def align_secondary(sec_hdf5,dem_file,topo_path,out_path,freqs=['A'],threshold = 1e-8,
                    numiter = 25, lines_per_block = 1000,sec_orbit = None,gpu_enabled = True,
                    gpu_id = 0):
    '''
    run rdr2geo
    '''
    out_path = pathlib.Path(out_path)
    slc = SLC(hdf5file=sec_hdf5)

    # Get orbit
    orbit = slc.getOrbit()
    if sec_orbit is not None:
        # SLC will get first radar grid whose frequency is available.
        # Reference epoch and orbit have no frequency dependency.
        external_orbit = load_orbit_from_xml(sec_orbit, slc.getRadarGrid().ref_epoch)
        orbit = crop_external_orbit(external_orbit, orbit)

    # Set ellipsoid based on DEM epsg
    dem_raster = isce3.io.Raster(str(dem_file))
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # NISAR RSLC products are always zero doppler
    doppler_grid = isce3.core.LUT2d()

    info_channel = journal.info('geo2rdr.run')
    info_channel.log("starting geo2rdr")

    # check if gpu use if required
    use_gpu = isce3.core.gpu_check.use_gpu(gpu_enabled,gpu_id)

    if use_gpu:
        # set CUDA device
        device = isce3.cuda.core.Device(gpu_id)
        isce3.cuda.core.set_device(device)

    t_all = time.time()

    for freq in freqs:

        # Get parameters specific for that frequency
        radar_grid = slc.getRadarGrid(frequency=freq)

        # Create geo2rdr directory
        geo2rdr_scratch_path = out_path / 'geo2rdr' / f'freq{freq}'
        geo2rdr_scratch_path.mkdir(parents=True, exist_ok=True)

        # Initialize CPU or GPU geo2rdr object accordingly
        if use_gpu:
            Geo2Rdr = isce3.cuda.geometry.Geo2Rdr
        else:
            Geo2Rdr = isce3.geometry.Geo2Rdr

        geo2rdr_obj = Geo2Rdr(radar_grid, orbit, ellipsoid, doppler_grid,
                              threshold, numiter, lines_per_block)

        # Open Topo Raster
        topo_path = pathlib.Path(topo_path)
        rdr2geo_topo_path = topo_path / 'rdr2geo' / f'freq{freq}' / 'topo.vrt'
        topo_raster = isce3.io.Raster(str(rdr2geo_topo_path))

        # Run geo2rdr
        geo2rdr_obj.geo2rdr(topo_raster, str(geo2rdr_scratch_path))

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran geo2rdr in {t_all_elapsed:.3f} seconds")


def extract_ref_topo(input_hdf5,dem_file,rdr2geo_cfg,out_path,freqs=['A'],threshold = 1e-8,
                    numiter = 25, extraiter = 10 ,lines_per_block = 1000,ref_orbit = None,
                    gpu_enabled = True,gpu_id = 0):


    out_path = pathlib.Path(out_path)
    slc = SLC(hdf5file=input_hdf5)

    # Get orbit
    orbit = slc.getOrbit()
    if ref_orbit is not None:
        # SLC will get first radar grid whose frequency is available.
        # Reference epoch and orbit have no frequency dependency.
        external_orbit = load_orbit_from_xml(ref_orbit, slc.getRadarGrid().ref_epoch)
        orbit = crop_external_orbit(external_orbit, orbit)

    # set defaults shared by both frequencies
    dem_raster = isce3.io.Raster(str(dem_file))
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # NISAR RSLC products are always zero doppler
    grid_doppler = isce3.core.LUT2d()

    info_channel = journal.info("rdr2geo.run")
    info_channel.log("starting rdr2geo")

    # check if gpu ok to use
    use_gpu = isce3.core.gpu_check.use_gpu(gpu_enabled,gpu_id)
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(gpu_id)
        isce3.cuda.core.set_device(device)

    t_all = time.time()
    for freq in freqs:
        # get frequency specific parameters
        radargrid = slc.getRadarGrid(freq)

        # create separate directory within scratch dir for rdr2geo run
        rdr2geo_scratch_path = out_path / 'rdr2geo' / f'freq{freq}'
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

        # rdr2geo_cfg = cfg['processing']['rdr2geo']

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