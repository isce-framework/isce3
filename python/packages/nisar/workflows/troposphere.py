#!/usr/bin/env python3
from datetime import datetime
import journal
import numpy as np
import os
import time

import pyaps3 as pa

import isce3
from isce3.core import transform_xy_to_latlon
from isce3.io import HDF5OptimizedReader
from nisar.workflows import h5_prep
from nisar.workflows.troposphere_runconfig import InsarTroposphereRunConfig
from nisar.products.insar.product_paths import GUNWGroupsPaths
from nisar.workflows.yaml_argparse import YamlArgparse


def compute_troposphere_delay(cfg: dict, gunw_hdf5: str):
    '''
    Compute the troposphere delay datacube

    Parameters
     ----------
     cfg: dict
        runconfig dictionary
     gunw_hdf5: str
        NISAR GUNW hdf5 file

    Returns
     -------
     troposphere_delay_datacube: dict
        troposphere delay datacube dictionary
    '''

    error_channel = journal.error('troposphere.compute_troposphere_delay')

    # Instantiate GUNW product object to avoid hard-coded paths to GUNW datasets
    gunw_obj = GUNWGroupsPaths()

    # Fetch the configurations
    tropo_weather_model_cfg = cfg['dynamic_ancillary_file_group']\
        ['troposphere_weather_model_files']
    tropo_cfg = cfg['processing']['troposphere_delay']

    scratch_path = cfg['product_path_group']['scratch_path']

    weather_model_type = tropo_cfg['weather_model_type'].upper()
    reference_weather_model_file = \
        tropo_weather_model_cfg['reference_troposphere_file']
    secondary_weather_model_file = \
        tropo_weather_model_cfg['secondary_troposphere_file']

    tropo_package = tropo_cfg['package'].lower()
    tropo_delay_direction = tropo_cfg['delay_direction'].lower()

    tropo_delay_products = []
    # comb is short for the summation of wet and dry components
    for delay_type in ['wet', 'hydrostatic', 'comb']:
        if tropo_cfg[f'enable_{delay_type}_product']:
            if (delay_type == 'hydrostatic') and \
                    (tropo_package == 'raider'):
                delay_type = 'hydro'
            if (delay_type == 'hydrostatic') and \
                    (tropo_package == 'pyaps'):
                delay_type = 'dry'

            tropo_delay_products.append(delay_type)

    # Troposphere delay datacube
    troposphere_delay_datacube = dict()

    with HDF5OptimizedReader(name=gunw_hdf5, mode='r', libver='latest', swmr=True) as h5_obj:

        # Fetch the GUWN Incidence Angle Datacube
        rdr_grid_path = gunw_obj.RadarGridPath

        inc_angle_cube = h5_obj[f'{rdr_grid_path}/incidenceAngle'][()]
        xcoord_radar_grid = h5_obj[f'{rdr_grid_path}/xCoordinates'][()]
        ycoord_radar_grid = h5_obj[f'{rdr_grid_path}/yCoordinates'][()]
        height_radar_grid = h5_obj[f'{rdr_grid_path}/heightAboveEllipsoid'][()]

        # EPSG code
        epsg = int(h5_obj[f'{rdr_grid_path}/projection'].attrs['epsg_code'])

        # Wavelenth in meters
        wavelength = isce3.core.speed_of_light / \
                h5_obj[f'{gunw_obj.GridsPath}/frequencyA/centerFrequency'][()]

        # X and y for the entire datacube
        y_2d_radar = np.tile(ycoord_radar_grid, (len(xcoord_radar_grid), 1)).T
        x_2d_radar = np.tile(xcoord_radar_grid, (len(ycoord_radar_grid), 1))

        # Lat/lon coordinates
        lat_datacube, lon_datacube, _ = transform_xy_to_latlon(
            epsg, x_2d_radar, y_2d_radar)

        # pyaps package
        if tropo_package == 'pyaps':

            for tropo_delay_product in tropo_delay_products:

                tropo_delay_datacube_list = []
                for index, hgt in enumerate(height_radar_grid):

                    dem_datacube = np.full(lat_datacube.shape, hgt)
                    # Delay for the reference image
                    ref_aps_estimator = pa.PyAPS(reference_weather_model_file,
                                                 dem=dem_datacube,
                                                 inc=0.0,
                                                 lat=lat_datacube,
                                                 lon=lon_datacube,
                                                 grib=weather_model_type,
                                                 humidity='Q',
                                                 model=weather_model_type,
                                                 verb=False,
                                                 Del=tropo_delay_product)

                    phs_ref = ref_aps_estimator.getdelay()

                    # Delay for the secondary image
                    second_aps_estimator = pa.PyAPS(secondary_weather_model_file,
                                                    dem=dem_datacube,
                                                    inc=0.0,
                                                    lat=lat_datacube,
                                                    lon=lon_datacube,
                                                    grib=weather_model_type,
                                                    humidity='Q',
                                                    model=weather_model_type,
                                                    verb=False,
                                                    Del=tropo_delay_product)

                    phs_second = second_aps_estimator.getdelay()

                    # Convert the delay in meters to radians
                    tropo_delay_datacube_list.append(
                            -(phs_ref - phs_second) * 4.0 * np.pi / wavelength)

                # Tropo delay datacube
                tropo_delay_datacube = np.stack(tropo_delay_datacube_list)
                tropo_delay_datacube_list = None

                if tropo_delay_direction == 'line_of_sight_mapping':
                    tropo_delay_datacube /= np.cos(np.deg2rad(inc_angle_cube))

                # Save to the dictionary in memory
                tropo_delay_product_name = \
                    f'tropoDelay_{tropo_package}_{tropo_delay_direction}_{tropo_delay_product}'
                troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube

        # raider package
        else:
            import xarray as xr
            import RAiDER
            from RAiDER.llreader import BoundingBox
            from RAiDER.losreader import Zenith, Raytracing
            from RAiDER.delay import tropo_delay as raider_tropo_delay
            from RAiDER.models.hres import HRES

            def _norm_lon(lon):
                '''
                Wrap longitudes to the [-180, 180) range.

                Parameters
                ----------
                lon: array_like
                    longitudes in degrees

                Returns
                -------
                    longitudes wrapped to [-180, 180)
                '''
                return ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0

            def _crosses_dateline(lon):
                '''
                Test whether a set of longitudes spans the antimeridian.

                Parameters
                ----------
                lon: array_like
                    longitudes in degrees

                Returns
                -------
                    True if the longitude span exceeds 180 deg (i.e. crosses +/-180)
                '''
                lon = _norm_lon(lon)
                return bool(np.nanmax(lon) - np.nanmin(lon) > 180.0)

            def _lon_per_xcolumn(lon_datacube):
                '''
                Representative longitude for each output x-column.

                Computed in a 0..360 frame when the cube straddles the seam so the
                per-column value is stable across +/-180. Used only to assign whole columns
                to a side (east or west); not used for the bounding box.

                Parameters
                ----------
                lon_datacube: np.ndarray
                    radar-grid longitudes in degrees, last axis = range/x

                Returns
                -------
                    1-D array of representative longitude per x-column
                '''
                lon = _norm_lon(lon_datacube)
                if np.nanmax(lon) - np.nanmin(lon) > 180.0:
                    lon = np.where(lon < 0, lon + 360.0, lon)
                return np.nanmean(lon, axis=tuple(range(lon.ndim - 1)))

            def _side_bounds(lat, lon, cols, side, margin):
                '''
                Geographic bounding box for a subset of output columns.

                For a split side the box is extended to +/-180 so the east and west halves
                meet at the seam; for side='all' it is the plain nan-safe min/max.

                Parameters
                ----------
                lat: np.ndarray
                    radar-grid latitudes in degrees, last axis = range/x
                lon: np.ndarray
                    radar-grid longitudes in degrees, last axis = range/x
                cols: np.ndarray
                    column indices belonging to this side
                side: str
                    one of 'west_dateline', 'east_dateline', or 'all'
                margin: float
                    bounding-box padding in degrees

                Returns
                -------
                    bounding box as [S, N, W, E] in degrees
                '''
                lat = np.take(np.asarray(lat, float), cols, axis=-1)
                lon = np.take(_norm_lon(lon), cols, axis=-1)
                s = float(np.nanmin(lat)) - margin
                n = float(np.nanmax(lat)) + margin
                if side == "west_dateline":                                # +lon hemisphere
                    pos = lon[np.isfinite(lon) & (lon >= 0)]
                    return [s, n, float(np.min(pos)) - margin, 180.0]
                if side == "east_dateline":                                # -lon hemisphere
                    neg = lon[np.isfinite(lon) & (lon < 0)]
                    return [s, n, -180.0, float(np.max(neg)) + margin]
                return [s, n, float(np.nanmin(lon)) - margin, float(np.nanmax(lon)) + margin]

            def compute_tropo_delay(dt, weather_model_file, lat_datacube, lon_datacube,
                                    xpts, ypts, los, height_levels, out_proj,
                                    weather_model_type=None,
                                    weather_model_output_dir=None,
                                    hres_converter=None, margin=0.1):
                '''
                Compute the RAiDER tropospheric delay with antimeridian handling.

                Splits the output columns into an east and a west sub-AOI only when the
                scene crosses +/-180, runs each against its own weather-model crop, and
                merges the two delay cubes back along x. When the scene does not cross the
                antimeridian it behaves like a single tropo_delay call.

                Parameters
                ----------
                dt: datetime
                    acquisition datetime, passed through to RAiDER
                weather_model_file: str
                    weather model file (RAiDER NetCDF, or HRES NetCDF if
                    weather_model_type == 'HRES')
                lat_datacube: np.ndarray
                    radar-grid latitudes in degrees, last axis = range/x
                lon_datacube: np.ndarray
                    radar-grid longitudes in degrees, last axis = range/x
                xpts: np.ndarray
                    output-grid x coordinates (xcoord_radar_grid)
                ypts: np.ndarray
                    output-grid y coordinates (ycoord_radar_grid)
                los: object
                    RAiDER line-of-sight object (e.g. Zenith or Raytracing)
                height_levels: list
                    height levels of the output delay cube
                out_proj: int or str
                    output projection (EPSG code) of the delay cube
                weather_model_type: str, optional
                    weather model name; if 'HRES', hres_converter is applied per sub-AOI
                weather_model_output_dir: str, optional
                    output directory for the RAiDER internal NetCDF (HRES conversion)
                hres_converter: callable, optional
                    function(weather_model_file, lat_lon_bounds, output_dir) -> file path,
                    used to convert an HRES NetCDF to the RAiDER internal NetCDF per sub-AOI
                margin: float, optional
                    AOI padding in degrees (default 0.1)

                Returns
                -------
                    the (merged) xarray.Dataset tropospheric delay cube
                '''

                xpts = np.asarray(xpts, float)
                ypts = np.asarray(ypts, float)

                def _run(bounds, xsub):
                    wm = weather_model_file
                    if weather_model_type == "HRES" and hres_converter is not None:
                        wm = hres_converter(weather_model_file, bounds,
                                            weather_model_output_dir)
                    aoi = BoundingBox(bounds)
                    aoi.xpts, aoi.ypts = xsub, ypts
                    return raider_tropo_delay(dt=dt, weather_model_file=wm,
                                              aoi=aoi, los=los,
                                              height_levels=height_levels,
                                              out_proj=out_proj)[0]

                # no crossing -> single call, original behaviour
                if not _crosses_dateline(lon_datacube):
                    bounds = _side_bounds(lat_datacube, lon_datacube,
                                        np.arange(xpts.size), "all", margin)
                    return _run(bounds, xpts)

                # crossing -> split columns by side, run each, merge along x by orig index
                rep = _lon_per_xcolumn(lon_datacube)
                merged = None
                for side, cols in (("west_dateline", np.where(rep <= 180.0)[0]),
                                ("east_dateline", np.where(rep > 180.0)[0])):
                    if cols.size == 0:
                        continue
                    bounds = _side_bounds(lat_datacube, lon_datacube, cols, side, margin)
                    td = _run(bounds, xpts[cols]).assign_coords(_xidx=("x", cols))
                    merged = td if merged is None else xr.concat([merged, td], dim="x")
                return merged.sortby("_xidx").drop_vars("_xidx")

            def _convert_HRES_to_raider_NetCDF(weather_model_file,
                                              lat_lon_bounds,
                                              weather_model_output_dir):
                '''
                Internal convenience function to convert the ECMWF NetCDF to RAiDER NetCDF

                Parameters
                ----------
                 weather_model_file: str
                    HRES NetCDF weather model file
                 lat_lon_bounds: list
                     bounding box of the RSLC
                 weather_model_output_dir: str
                     the output directory of the RAiDER internal NetCDF file
                Returns
                -------
                     the path of the RAiDER internal NetCDF file
                 '''

                os.makedirs(weather_model_output_dir, exist_ok=True)
                ds = xr.open_dataset(weather_model_file)

                # Get the datetime of the weather model file
                weather_model_time = \
                    ds.time.values.astype('datetime64[s]').astype(datetime)[0]
                hres = HRES()
                # Set up the time, Lat/Lon, and working location, where
                # the lat/lon bounds are applied to clip the global
                # weather model to minimize the data processing
                hres.setTime(weather_model_time)

                # Workaround for a RAiDER bug in the version of '0.5.2' and '0.5.3'
                # (see https://github.com/dbekaert/RAiDER/issues/682)
                if RAiDER.__version__ in ['0.5.2','0.5.3']:
                    hres._time = hres._time.replace(tzinfo=None)

                hres.set_latlon_bounds(ll_bounds = lat_lon_bounds)
                hres.set_wmLoc(weather_model_output_dir)

                # Load the ECMWF NetCDF weather model
                hres.load_weather(weather_model_file)

                # Process the weather model data
                hres._find_e()
                hres._uniform_in_z(_zlevels=None)

                # This function implemented in the RAiDER
                # fills the NaNs with 0
                hres._checkForNans()

                hres._get_wet_refractivity()
                hres._get_hydro_refractivity()
                hres._adjust_grid(hres.get_latlon_bounds())

                # Compute Zenith delays at the weather model grid nodes
                hres._getZTD()

                output_file = hres.out_file(weather_model_output_dir)
                hres._out_name =  output_file

                # Return the output file if it exists
                if os.path.exists(output_file):
                    return output_file
                else:
                    # Write to hard drive
                    return hres.write()

            # output location
            weather_model_output_dir = \
                os.path.join(scratch_path, 'weather_model_files')

            # Acquisition time for reference and secondary images
            acquisition_time_ref = h5_obj[f'{gunw_obj.IdentificationPath}/referenceZeroDopplerStartTime'][()]\
                    .astype('datetime64[s]').astype(datetime)
            acquisition_time_second = h5_obj[f'{gunw_obj.IdentificationPath}/secondaryZeroDopplerStartTime'][()]\
                    .astype('datetime64[s]').astype(datetime)

            # Zenith
            delay_direction_obj = Zenith()

            if tropo_delay_direction == 'line_of_sight_raytracing':
                delay_direction_obj = Raytracing()

            # Height levels
            height_levels = list(height_radar_grid)

            # Copmute the tropo delay with dateline safe
            tropo_delay_reference = compute_tropo_delay(
                dt=acquisition_time_ref,
                weather_model_file=reference_weather_model_file,
                lat_datacube=lat_datacube, lon_datacube=lon_datacube,
                xpts=xcoord_radar_grid, ypts=ycoord_radar_grid,
                los=delay_direction_obj, height_levels=height_levels, out_proj=epsg,
                weather_model_type=weather_model_type,
                weather_model_output_dir=weather_model_output_dir,
                hres_converter=_convert_HRES_to_raider_NetCDF)

            tropo_delay_secondary = compute_tropo_delay(
                dt=acquisition_time_second,
                weather_model_file=secondary_weather_model_file,
                lat_datacube=lat_datacube, lon_datacube=lon_datacube,
                xpts=xcoord_radar_grid, ypts=ycoord_radar_grid,
                los=delay_direction_obj, height_levels=height_levels, out_proj=epsg,
                weather_model_type=weather_model_type,
                weather_model_output_dir=weather_model_output_dir,
                hres_converter=_convert_HRES_to_raider_NetCDF)

            for tropo_delay_product in tropo_delay_products:
                # Compute troposphere delay with raider package
                # comb is the summation of wet and hydro components
                if tropo_delay_product == 'comb':
                    tropo_delay = \
                        tropo_delay_reference['wet'] + tropo_delay_reference['hydro'] - \
                            tropo_delay_secondary['wet'] - tropo_delay_secondary['hydro']
                else:
                    tropo_delay = tropo_delay_reference[tropo_delay_product] - \
                            tropo_delay_secondary[tropo_delay_product]

                # Convert it to radians units
                tropo_delay_datacube = -tropo_delay * 4.0 * np.pi / wavelength

                # Line of sight mapping
                if tropo_delay_direction == 'line_of_sight_mapping':
                    tropo_delay_datacube /= np.cos(np.deg2rad(inc_angle_cube))

                # Save to the dictionary in memory
                tropo_delay_product_name = \
                    f'tropoDelay_{tropo_package}_{tropo_delay_direction}_{tropo_delay_product}'
                troposphere_delay_datacube[tropo_delay_product_name]  = tropo_delay_datacube

    return troposphere_delay_datacube


def write_to_GUNW_product(tropo_delay_datacubes: dict, gunw_hdf5: str):
    '''
    Write the troposphere delay datacubes to GUNW product

    Parameters
     ----------
     tropo_delay_datacubes: dict
        troposphere delay datacube dictionary
      gunw_hdf5: str
         gunw hdf5 file

    Returns
     -------
       None
    '''
    # Instantiate GUNW object to avoid hard-coded path to GUNW datasets
    gunw_obj = GUNWGroupsPaths()
    with HDF5OptimizedReader(name=gunw_hdf5, mode='a', libver='latest', swmr=True) as f:

        for product_name, product_cube in tropo_delay_datacubes.items():

             radar_grid = f.get(gunw_obj.RadarGridPath)

             # Troposphere delay product information
             products = product_name.split('_')
             package = products[1]
             delay_product = products[-1]
             delay_direction = products[2:-1]

             # Delay product
             delay_product = delay_product.lower()

             if delay_product == 'comb':
                 delay_product = 'combined'

             # The 'hydro' term is used by radier package for the dry component,
             # The 'dry' term is used by the pyaps package for the dry comopnent
             # NISAR uses 'hydrostatic' to describe the dry component
             if delay_product in ['hydro', 'dry']:
                 delay_product = 'hydrostatic'

             # Delay direction
             delay_direction = '_'.join(delay_direction).lower()

             if delay_direction == 'line_of_sight_mapping':
                 delay_direction = 'Lineofsight'
             elif delay_direction == 'line_of_sight_raytracing':
                 delay_direction = 'Raytracing'
             else:
                 delay_direction = 'Zenith'

             # Troposphere delay Package
             tropo_pkg = package.lower()

             # pyAPS
             if tropo_pkg == 'pyaps':
                 tropo_pkg = 'pyAPS'
             # RAiDER
             if tropo_pkg == 'raider':
                 tropo_pkg = 'RAiDER'

             # Dataset description
             descr = f"{delay_product.capitalize()} component of the troposphere phase screen"

             # Product name
             output_product_name = f'{delay_product}TroposphericPhaseScreen'

             # If there is no troposphere delay product, then create a new one
             if output_product_name not in radar_grid:
                 h5_prep._create_datasets(radar_grid, [0], np.float64,
                                          output_product_name, descr = descr,
                                          units='radians',
                                          data=product_cube.astype(np.float64))

             # If there exists the product, overwrite the old one
             else:
                 tropo_delay = radar_grid[output_product_name]
                 tropo_delay[:] = product_cube.astype(np.float64)

        f.close()

def run(cfg: dict, gunw_hdf5: str):
    '''
    compute the troposphere delay and write to GUNW product

    Parameters
     ----------
     cfg: dict
        runconfig dictionary
     gunw_hdf5: str
        gunw hdf5 file

    Returns
     -------
        None
    '''

    # Create error and info channels
    info_channel = journal.info("troposphere.run")
    info_channel.log("starting insar_troposphere_delay computation")

    t_all = time.time()

    # Compute the troposphere delay datacube
    tropo_delay_datacube = compute_troposphere_delay(cfg, gunw_hdf5)

    # Write to GUNW product
    write_to_GUNW_product(tropo_delay_datacube, gunw_hdf5)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran troposphere delay in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":

    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    tropo_runcfg = InsarTroposphereRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(tropo_runcfg.cfg)
    run(tropo_runcfg.cfg, gunw_hdf5 = out_paths['GUNW'])
