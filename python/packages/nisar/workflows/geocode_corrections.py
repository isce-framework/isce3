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

from nisar.products.writers.SLC import time_units
from nisar.workflows.helpers import add_dataset_and_attrs, HDF5DatasetParams


def _get_decimated_radar_grid(radar_grid_orig,
                              orbit,
                              pysolid_posting=5000.0):
    '''Helper function to decimate original full resolution radar grid down to
    5km resolution in az and slant range.

    5km is optimal resolution according to pySolid documentation.

    Parameters
    ----------
    radar_grid_orig: isce3.product.RadarGridParameters
        Radargrid of the input RSLC
    orbit: isce3.core.Orbit
        Orbit of the input RSLC
    pysolid_posting: float, Default 5000.0
        Posting of the solid earth tide computation using PySolid in meters

    Returns
    -------
    radar_grid_scaled: isce3.product.RadarGridParameters
    '''

    # Get azimuth and ground range spacing in meters.
    azimuth_spacing, ground_range_spacing = \
        get_radar_grid_nominal_ground_spacing(radar_grid_orig, orbit)

    # Compute scaling factor needed for 5km azimuth resolution in radar grid.
    az_scaling_factor = azimuth_spacing / pysolid_posting

    # Compute scaled length based on computed azimuth scaling factor.
    length_scaled = max(int(az_scaling_factor * radar_grid_orig.length), 2)

    # Compute scaling factor needed for 5km slant range resolution in radar grid.
    srg_scaling_factor = ground_range_spacing / pysolid_posting

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
     Reading done separately as GDAL does not flush buffers before the end of computation.
    '''
    # Compute lat and lon for scaled radar grid pixels. To be used for
    # interpolating SET.
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # NISAR RSLC products are always zero Doppler
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
        Str identification for NISAR SLC frequencies
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
        Str identification for NISAR SLC frequencies
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

    # DEM file for DEM interpolator and EPSG for ellipsoid
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']

    tec_correction = tec_lut2d_from_json_srg(tec_file, center_freq, orbit,
                                             radar_grid, doppler, dem_file)

    return tec_correction


class AzSrgCorrections:
    '''
    Class to compute, store, and write to HDF5 model and data driven corrections.
    '''
    def __init__(self, cfg, slc, frequency, orbit):
        '''
        Compute azimuth and slant range geocoding corrections.

        Default to zero-valued LUT2d for either if provided parameters do not require
        corrections to be computed.

        Parameters
        ----------
        cfg: dict
            Dict containing the runconfiguration parameters
        slc: nisar.products.readers.SLC
            NISAR single look complex (SLC) object containing swath and radar grid
            parameters
        frequency: ['A', 'B']
            Str identification for NISAR SLC frequencies
        orbit: isce3.core.Orbit
            Object containing orbit associated with SLC
        '''
        # Dict with azimuth correction name key and value as
        # corresponding array of correction values
        self.az_correction_arrays = {}

        # Cumulative azimuth correction LUT2d
        self.az_correction_lut = isce3.core.LUT2d()

        # Dict with slant range correction name key and value as
        # corresponding array of correction values
        self.slant_range_correction_arrays = {}

        # Cumulative slant range correction LUT2d
        self.slant_range_correction_lut = isce3.core.LUT2d()

        # Save following as instance members for convenience
        self.cfg = cfg
        self.slc = slc
        self.frequency = frequency
        self.orbit = orbit

        # Decimate radar grid to 5km resolution in azimuth and slant range
        radar_grid = self.slc.getRadarGrid(self.frequency)
        self.radar_grid_scaled = _get_decimated_radar_grid(radar_grid,
                                                           self.orbit)

        # Using decimated radar grid, compute axis of grid to upsample TEC array to
        self.az_vec = self.radar_grid_scaled.sensing_start + \
            np.arange(self.radar_grid_scaled.length) / self.radar_grid_scaled.prf
        self.rg_vec = self.radar_grid_scaled.starting_range + \
            np.arange(self.radar_grid_scaled.width) * self.radar_grid_scaled.range_pixel_spacing

        # Unpack flags and determine which corrections to generate
        self.correct_set = cfg['processing']['correction_luts']['solid_earth_tides_enabled']
        self.correct_tec = cfg["dynamic_ancillary_file_group"]['tec_file'] is not None
        if self.correct_set or self.correct_tec:
            self._compute_model_correction_luts()

        # GSLC only - Check for reference GSLC to compute data driven corrections
        if 'reference_gslc' in cfg['dynamic_ancillary_file_group']:
            self.apply_data_driven_correction = cfg['dynamic_ancillary_file_group']['reference_gslc'] is not None
        else:
            self.apply_data_driven_correction = False

        if self.apply_data_driven_correction:
            self._compute_offset_luts()


    def _compute_offset_luts(self):
        '''
        A placeholder to compute timing correction based on offset tracking (ampcor)
        '''
        info_channel = journal.info("AzSrgCorrections._compute_offset_luts")

        # TODO future ISCE3 release will need to add:
        # azimuth and slant items into az_correction_arrays and
        # slant_range_correction_arrays
        # add update azimuth and slant range correction LUTs with offset arrays
        info_channel.log('Data-driven timing correction for GSLC is not implemented.')


    def _compute_model_correction_luts(self):
        '''
        Compute TEC and/or SET corrections. Add the results to  azimuth /
        slant range correction arrays.
        '''
        # If no corrections to be generated, default LUT2d for azimuth and slant range will be untouched
        # If flagged, generate Solid Earth Tide (SET) corrections array for slant range only

        warning_channel = journal.warning("geocode_corrections._compute_model_correction_luts")

        if self.correct_set:
            # Prepare inputs for computing decimated geogrid to interpolated to.
            dem_raster = isce3.io.Raster(self.cfg['dynamic_ancillary_file_group']['dem_file'])
            epsg = dem_raster.get_epsg()
            proj = isce3.core.make_projection(epsg)
            ellipsoid = proj.ellipsoid

            # Compute latitude and longitude over decimated radar grid
            scratch_path = pathlib.Path(self.cfg['product_path_group']['scratch_path']) / "geocode_corrections"
            scratch_path.mkdir(parents=True, exist_ok=True)
            _compute_llh_coords(self.cfg,
                                self.radar_grid_scaled,
                                dem_raster,
                                self.orbit,
                                scratch_path)
            (x_pts_to_interp,
             y_pts_to_interp,
             z_pts_to_interp)= _read_llh(scratch_path)

            # Compute solid earth tides over decimated/scaled radar grid
            set_rg, _ = solid_earth_tides(self.radar_grid_scaled,
                                          x_pts_to_interp,
                                          y_pts_to_interp,
                                          z_pts_to_interp,
                                          self.orbit,
                                          ellipsoid)

        # If flagged, generate very low res TEC LUT2d for azimuth and slant range.
        if self.correct_tec:
            low_res_tec_az = _get_iono_azimuth_corrections(self.cfg,
                                                           self.slc,
                                                           self.frequency,
                                                           self.orbit)
            low_res_tec_srange = _get_iono_srange_corrections(self.cfg,
                                                              self.slc,
                                                              self.frequency,
                                                              self.orbit)

            # If only TEC corrections generated, return existing TEC correction LUT2ds
            if not self.correct_set:
                self.az_correction_lut = low_res_tec_az
                self.az_correction_arrays["TEC"] = low_res_tec_az.data
                self.slant_range_correction_lut = low_res_tec_srange
                self.slant_range_correction_arrays["TEC"] = low_res_tec_srange.data

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
        if not self.correct_tec and self.correct_set:
            self.slant_range_correction_arrays["SET"] = set_rg
            self.slant_range_correction_lut = \
                    _make_correction_LUT2d(self.radar_grid_scaled, set_rg)

        # If TEC and SET corrections computed, upsample TEC to decimated radar grid resolution.
        # Then generate corresponding LUT2d's
        elif self.correct_tec and self.correct_set:
            def _eval_lut2d(lut2d, az_vec, rg_vec):
                # Helper function to evaluate low res TEC data to resolution of SET
                out_shape = (len(az_vec), len(rg_vec))
                arr = np.array([lut2d.eval(az, rg)
                                for az, rg in itertools.product(az_vec, rg_vec)])
                arr = arr.reshape(out_shape)
                return arr

            for which_lut, low_res_tec_lut2d in zip(('azimuth TEC correction', 'range TEC correction'),
                                                (low_res_tec_az, low_res_tec_srange)):
                lut2d_far_range = low_res_tec_lut2d.x_start + (low_res_tec_lut2d.width - 1) * low_res_tec_lut2d.x_spacing
                if self.rg_vec[-1] != lut2d_far_range:
                    warning_channel.log('Truncation error detected between '
                                        f'far range of scaled radargrid and {which_lut}. '
                                        f'Difference = ({lut2d_far_range - self.rg_vec[-1]}). '
                                        'bounds_error in the LUT turned off.')
                    low_res_tec_lut2d.bounds_error=False

            tec_az, tec_rg = [
                _eval_lut2d(low_res_tec_lut2d,
                            self.az_vec,
                            self.rg_vec)
                for low_res_tec_lut2d in [low_res_tec_az, low_res_tec_srange]]
            self.az_correction_arrays["TEC"] = tec_az
            self.slant_range_correction_arrays["TEC"] = tec_rg

            # Use only use TEC for azimuth
            az_correction_arrays_arr = tec_az
            # Use TEC and SET for slant range
            srange_corrections_arr = set_rg + tec_rg
            self.slant_range_correction_arrays["SET"] = set_rg

            (self.az_correction_lut,
                    self.slant_range_correction_lut) = (
                        _make_correction_LUT2d(self.radar_grid_scaled, data)
                        for data in (az_correction_arrays_arr,
                                     srange_corrections_arr))


    def write_corrections_hdf5(self, dst_parent_group):
        '''Write stored correction arrays to metadata h5py group.

        Parameters
        ----------
        dst_parent_group: h5py.Group
            h5py group to write corrections to
        '''
        if not self.az_correction_arrays and not self.slant_range_correction_arrays:
            # no LUTs to write
            return

        correction_group = dst_parent_group.require_group(f'timingCorrections/frequency{self.frequency}')

        # correction LUTs axis and Doppler correction LUTs
        units_time = np.bytes_(time_units(self.radar_grid_scaled.ref_epoch))
        correction_items = [
            HDF5DatasetParams(name='slantRange',
                              value=self.rg_vec,
                              description=('Slant range dimension corresponding to the'
                                           ' timing correction lookup tables'),
                              attr_dict={'units': 'meters'}),
            HDF5DatasetParams(name='slantRangeSpacing',
                              value=self.az_correction_lut.x_spacing,
                              description=('Slant range spacing of the '
                                           'timing correction lookup tables'),
                              attr_dict={'units': 'meters'}),
            HDF5DatasetParams(name='zeroDopplerTime',
                              value=self.az_vec,
                              description=('Zero Doppler time dimension since '
                                           'UTC epoch corresponding to the '
                                           'timing correction lookup tables'),
                              attr_dict={'units': units_time}),
            HDF5DatasetParams(name='zeroDopplerTimeSpacing',
                              value=self.az_correction_lut.y_spacing,
                              description=('Time interval in the along-track direction '
                                           'of the timing correction lookup tables'),
                              attr_dict={'units': 'seconds'})
            ]
        for az_rg_corr, az_rg_str, units in zip([self.az_correction_arrays,
                                                 self.slant_range_correction_arrays],
                                                ['azimuth', 'slantRange'],
                                                [units_time, 'meters']):

            az_or_los = 'azimuth' if az_rg_str=='azimuth' else 'line-of-sight'

            # TODO future release make accommodations for data driven corrections
            if "TEC" in az_rg_corr:
                what_correction = 'ionosphere timing correction'
                derived_from = ' derived from Total Electron Content data'
                correction_items.append(
                    HDF5DatasetParams(name=f'{az_rg_str}Ionosphere',
                                      value=az_rg_corr["TEC"],
                                      description=f'2D lookup table of {az_or_los} {what_correction}{derived_from}',
                                      attr_dict={'units': units}))
            if "SET" in az_rg_corr:
                what_correction = 'solid Earth tides timing correction'
                derived_from = ''
                correction_items.append(
                    HDF5DatasetParams(name=f'{az_rg_str}SolidEarthTides',
                                      value=az_rg_corr["SET"],
                                      description=f'2D lookup table of {az_or_los} {what_correction}{derived_from}',
                                      attr_dict={'units': units}))
        for meta_item in correction_items:
            add_dataset_and_attrs(correction_group, meta_item)


def get_az_srg_corrections(cfg, slc, frequency, orbit):
    '''
    A wrapper function to compute azimuth and slant range geocoding corrections
    and return as LUT2d.
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
        Str identification for NISAR SLC frequencies
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Yields
    ------
    az_correction_lut: isce3.core.LUT2d
        Azimuth correction for geocoding. Unit in seconds.
    slant_range_correction_lut: isce3.core.LUT2d
        Slant range correction for geocoding. Unit in meters.
    '''
    corrs = AzSrgCorrections(cfg, slc, frequency, orbit)
    return corrs.az_correction_lut, corrs.slant_range_correction_lut
