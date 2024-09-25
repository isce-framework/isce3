'''
Compute azimuth and slant range geocoding corrections as LUT2d
'''
import itertools
import pathlib

import numpy as np
from osgeo import gdal

import isce3
import journal
from isce3.atmosphere.tec_product import (tec_lut2d_from_json_srg,
                                          tec_lut2d_from_json_az)
from isce3.product import get_radar_grid_nominal_ground_spacing
from isce3.solid_earth_tides import solid_earth_tides


def _get_decimated_radar_grid(radar_grid_orig,
                              orbit):
    '''Helper function to decimate original full resolution radar grid down to
    5km resolution in az and slant range.
    '''
    # 5km is optimal resolution according to pySolid documentation.
    optimal_pysolid_res = 5000.0

    # Get azimuth and ground range spacing in meters.
    azimuth_spacing, ground_range_spacing = \
        get_radar_grid_nominal_ground_spacing(radar_grid_orig, orbit)

    # Compute scaling factor needed for 5km azimuth resolution in radar grid.
    az_scaling_factor = azimuth_spacing / optimal_pysolid_res

    # Compute scaled length based on computed azimuth scaling factor.
    length_scaled = max(int(az_scaling_factor * radar_grid_orig.length), 2)

    # Compute scaling factor needed for 5km slant range resolution in radar grid.
    srg_scaling_factor = ground_range_spacing / optimal_pysolid_res

    # Compute scaled width based on computed slant range scaling factor.
    width_scaled = max(int(srg_scaling_factor * radar_grid_orig.width), 2)

    # Resize radar grid while preserving start and stop.
    radar_grid_scaled = \
        radar_grid_orig.resize_and_keep_startstop(length_scaled, width_scaled)

    return radar_grid_scaled


def _compute_llh_coords(cfg,
                        radar_grid,
                        dem_raster,
                        orbit,
                        scratch_path):
    '''Compute the latitude and longitude of radar grid pixels.
     Reading done separately as GDAL does not flush buffers before the end of computaton.
    '''
    # Compute lat and lon for scaled radar grid pixels. To be used for
    # interpolating SET.
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # NISAR RSLC products are always zero doppler
    doppler_grid = isce3.core.LUT2d()

    # check if gpu ok to use
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                           cfg['worker']['gpu_id'])
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)

    # init CPU or CUDA object accordingly
    if use_gpu:
        Rdr2Geo = isce3.cuda.geometry.Rdr2Geo
    else:
        Rdr2Geo = isce3.geometry.Rdr2Geo
    rdr2geo_obj = Rdr2Geo(radar_grid,
                          orbit,
                          ellipsoid,
                          doppler_grid,
                          threshold=1.e-7)

    # Prepare x, y, and z output rasters
    fnames = "xyz"
    xyz_rasters = [isce3.io.Raster(f"{str(scratch_path)}/{fname}.rdr",
                                   radar_grid.width,
                                   radar_grid.length,
                                   1,
                                   gdal.GDT_Float64,
                                   "GTiff",
        )
        for fname in fnames
    ]

    # Run topo
    none_rasters = [None] * 8
    rdr2geo_obj.topo(
        dem_raster,
        *xyz_rasters,
        *none_rasters
    )


def _read_llh(scratch_path):
    # Read x, y, z, incidence, and heading/azimuth to arrays.
    def _gdal_raster_to_array(raster_path):
        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds.FlushCache()
        del ds
        return arr

    # XXX These are useless since GDAL has flushed values to disk yet. =/
    x, y, z = [
        _gdal_raster_to_array(f"{scratch_path}/{fname}.rdr")
        for fname in "xyz"
    ]

    # If DEM EPSG not 4326 convert rdr2geo output to it
    return x, y, z


def _get_iono_azimuth_corrections(cfg, slc, frequency, orbit):
    '''
    Compute and return TEC geolocation corrections for azimuth as a LUT2d.

    Parameters
    ----------
    cfg: dict
        Dict containing the runconfiguration parameters
    slc: nisar.products.readers.SLC
        NISAR single look complex (SLC) object containing swath and radar grid
        parameters
    frequency: ['A', 'B']
        Str identifcation for NISAR SLC frequencies
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Returns
    -------
    tec_correction: isce3.core.LUT2d
        TEC azimuth correction LUT2d for geocoding.
    '''
    # Compute TEC slant range correction if TEC file is provided
    tec_file = cfg["dynamic_ancillary_file_group"]['tec_file']

    center_freq = slc.getSwathMetadata(frequency).processed_center_frequency
    radar_grid = slc.getRadarGrid(frequency)

    tec_correction = tec_lut2d_from_json_az(tec_file, center_freq, orbit,
                                            radar_grid)

    return tec_correction


def _get_iono_srange_corrections(cfg, slc, frequency, orbit):
    '''
    Compute and return TEC corrections for slant range as LUT2d.

    Currently on TEC corrections available. Others will be added as they
    become available.

    Parameters
    ----------
    cfg: dict
        Dict containing the runconfiguration parameters
    slc: nisar.products.readers.SLC
        NISAR single look complex (SLC) object containing swath and radar grid
        parameters
    frequency: ['A', 'B']
        Str identifcation for NISAR SLC frequencies
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Yields
    ------
    tec_correction: isce3.core.LUT2d
        Slant range correction for geocoding. Currently only TEC corrections
        are considered. If no TEC JSON file is provided in the cfg parameter,
        a default isce3.core.LUT2d will be passed back.
    '''
    # Compute TEC slant range correction if TEC file is provided
    tec_file = cfg["dynamic_ancillary_file_group"]['tec_file']

    center_freq = slc.getSwathMetadata(frequency).processed_center_frequency
    doppler = isce3.core.LUT2d()
    radar_grid = slc.getRadarGrid(frequency)

    # DEM file for DEM interpolator and ESPF for ellipsoid
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']

    tec_correction = tec_lut2d_from_json_srg(tec_file, center_freq, orbit,
                                             radar_grid, doppler, dem_file)

    return tec_correction


def get_az_srg_corrections(cfg, slc, frequency, orbit):
    '''
    Compute azimuth and slant range geocoding corrections and return as LUT2d.
    Default to default LUT2d for either if provided parameters do not require
    corrections to be computed.

    Parameters
    ----------
    cfg: dict
        Dict containing the runconfiguration parameters
    slc: nisar.products.readers.SLC
        NISAR single look complex (SLC) object containing swath and radar grid
        parameters
    frequency: ['A', 'B']
        Str identifcation for NISAR SLC frequencies
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Yields
    ------
    az_corrections: isce3.core.LUT2d
        Azimuth correction for geocoding. Unit in seconds.
    srange_corrections: isce3.core.LUT2d
        Slant range correction for geocoding. Unit in meters.
    '''

    warning_channel = journal.warning("geocode_corrections.get_az_srg_corrections")
    # Unpack flags and determine which corrections to generate
    correct_set = cfg['processing']['correction_luts']['solid_earth_tides_enabled']
    correct_tec = cfg['dynamic_ancillary_file_group']['tec_file'] is not None

    # If no corrections to be generated, return default LUT2d for azimuth and slant range
    if not correct_set and not correct_tec:
        return [isce3.core.LUT2d()] * 2

    # If flagged, generate Solid Earth Tide (SET) corrections array for slant range only
    if correct_set:
        # Prepare inputs for computing decimated geogrid to interpolated to.
        dem_raster = isce3.io.Raster(cfg['dynamic_ancillary_file_group']['dem_file'])
        epsg = dem_raster.get_epsg()
        proj = isce3.core.make_projection(epsg)
        ellipsoid = proj.ellipsoid
        radar_grid = slc.getRadarGrid(frequency)

        # Decimate radar grid to 5km resolution in azimuth and slant range
        radar_grid_scaled = _get_decimated_radar_grid(radar_grid, orbit)

        # Compute latitude and longitude over decimated radar grid
        scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path']) / "geocode_corrections"
        scratch_path.mkdir(parents=True, exist_ok=True)
        _compute_llh_coords(cfg,
                            radar_grid_scaled,
                            dem_raster,
                            orbit,
                            scratch_path)
        (x_pts_to_interp,
         y_pts_to_interp,
         z_pts_to_interp)= _read_llh(scratch_path)

        # Compute solit earth tides over decimated/scaled radar grid
        set_rg, _ = solid_earth_tides(radar_grid_scaled,
                                      x_pts_to_interp,
                                      y_pts_to_interp,
                                      z_pts_to_interp,
                                      orbit,
                                      ellipsoid)

    # If flagged, generate very low res TEC LUT2d for azimuth and slant range.
    if correct_tec:
        low_res_tec_az = _get_iono_azimuth_corrections(cfg, slc, frequency,
                                                       orbit)
        low_res_tec_srange = _get_iono_srange_corrections(cfg, slc, frequency,
                                                          orbit)

    # If only TEC corrections generated, return existing TEC correction LUT2ds
    if correct_tec and not correct_set:
        return low_res_tec_az, low_res_tec_srange

    def _make_correction_LUT2d(radar_grid, data):
        # Helper function to make LUT2d from correction arrays
        return isce3.core.LUT2d(radar_grid.starting_range,
                                radar_grid.sensing_start,
                                radar_grid.range_pixel_spacing,
                                1 / radar_grid.prf,
                                data)
    # If only SET range correction generated, return
    # 1. populated slant range LUT2d
    # 2. default LUT2d for azimuth i.e. no corrections in azimuth
    if not correct_tec and correct_set:
        az_lut = isce3.core.LUT2d()
        srange_lut = _make_correction_LUT2d(radar_grid_scaled, set_rg)

    # If TEC and SET corrections computed, upsample TEC to radar grid resolution.
    # Then generate corresponding LUT2d's
    if correct_tec and correct_set:
        # Using decimated radar grid, compute axis of grid to upsample TEC array to
        az_vec = radar_grid_scaled.sensing_start + \
            np.arange(radar_grid_scaled.length) / radar_grid_scaled.prf
        rg_vec = radar_grid_scaled.starting_range + \
            np.arange(radar_grid_scaled.width) * radar_grid_scaled.range_pixel_spacing
        
        # Check if the last elements in `rg_vec` have truncation error
        for which_lut, low_res_tec_lut2d in zip(('azimuth TEC correction', 'range TEC correction'),
                                                (low_res_tec_az, low_res_tec_srange)):
            lut2d_far_range = low_res_tec_lut2d.x_start + (low_res_tec_lut2d.width - 1) * low_res_tec_lut2d.x_spacing
            if rg_vec[-1] != lut2d_far_range:
                warning_channel.log('Truncation error detected between '
                                    f'far range of scaled radargrid and {which_lut}. '
                                    f'Difference = ({lut2d_far_range - rg_vec[-1]}). '
                                    'bounds_error in the LUT turned off.')
                low_res_tec_lut2d.bounds_error=False
            
        


        def _eval_lut2d(lut2d, az_vec, rg_vec, out_shape):
            # Helper function to evaluate low res TEC data to resolution of SET
            arr = np.array([lut2d.eval(az, rg)
                            for az, rg in itertools.product(az_vec, rg_vec)])
            arr = arr.reshape(out_shape)
            return arr

        tec_az, tec_rg = [
            _eval_lut2d(low_res_tec_lut2d,
                        az_vec,
                        rg_vec,
                        radar_grid_scaled.shape)
            for low_res_tec_lut2d in [low_res_tec_az, low_res_tec_srange]]

        # Use only use TEC for azimuth
        az_corrections_arr = tec_az
        # Use TEC and SET for slant range
        srange_corrections_arr = set_rg + tec_rg

        az_lut, srange_lut = [
            isce3.core.LUT2d(radar_grid_scaled.starting_range,
                             radar_grid_scaled.sensing_start,
                             radar_grid_scaled.range_pixel_spacing,
                             1 / radar_grid_scaled.prf,
                             data)
            for data in [az_corrections_arr, srange_corrections_arr]]

    return az_lut , srange_lut


def get_offset_luts(cfg, slc, frequency, orbit):
    '''
    A placeholder to compute timing correction based on offset tracking (ampcor)

    Parameters
    ----------
    cfg: dict
    frequency: ['A', 'B']
        Str identifcation for NISAR SLC frequencies
    slc: nisar.products.readers.SLC
        NISAR single look complex (SLC) object containing swath and radar grid
        parameters
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Returns
    -------
    az_lut: isce3.core.LUT2d
        2d LUT in azimuth time (seconds) for geolocation correction in azimuth direction.
    rg_lut: isce3.core.LUT2d
        2d LUT in meters for geolocation correction in slant range direction.
    '''
    info_channel = journal.info("geocode_corrections.get_offset_lut")

    info_channel.log('Data-driven GSLC will be implemented in the next release.'
                     ' Currently returning empty LUT2d of timing corrections in'
                     ' both range and azimuth directions.')

    rg_lut = isce3.core.LUT2d()
    az_lut = isce3.core.LUT2d()
    return az_lut, rg_lut
