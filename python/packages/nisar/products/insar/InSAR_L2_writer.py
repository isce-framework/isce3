import isce3
import numpy as np
from isce3.core import LUT2d
from nisar.workflows.h5_prep import (_get_raster_from_hdf5_ds,
                                     add_radar_grid_cubes_to_hdf5,
                                     set_get_geo_info)
from nisar.workflows.helpers import get_cfg_freq_pols

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_base_writer import InSARBaseWriter
from .InSAR_L1_writer import L1InSARWriter
from .product_paths import L2GroupsPaths
from .units import Units
from .utils import extract_datetime_from_string


class L2InSARWriter(L1InSARWriter):
    """
    Writer class for L2InSARWriter products (GOFF and GUNW)
    inherent from L1InSARWriter
    """
    def __init__(self, **kwds):
        """
        Constructor for L2InSARWriter class
        """
        super().__init__(**kwds)

        # group paths are Level 2 group paths
        self.group_paths = L2GroupsPaths()

    def save_to_hdf5(self):
        """
        Save to HDF5
        """
        InSARBaseWriter.save_to_hdf5(self)

        self.add_radar_grid_cubes()
        self.add_grids_to_hdf5()

    def add_secondary_radar_grid_cube(self, sec_cube_group_path,
                                       geogrid, heights, radar_grid, orbit,
                                       native_doppler, grid_doppler,
                                       threshold_geo2rdr=1e-8,
                                       numiter_geo2rdr=100,
                                       delta_range=1e-8):
        """
        Add the slant range and azimuth time cubes of the secondary image to the
        radar grids

        Parameters
        ----------
        sec_cube_group_path : str
            Radar grid cube path of the secondary image
        geogrid : isce3.product.GeoGridParameters
            Geogrid object
        heights : list
            The radar grid cube heights
        radar_grid : isce3.product.RadarGridParameters
            The radar grid object
        orbit : isce3.core.Orbit
            The orbit object
        native_doppler : isce3.core.LUT2d
            The native doppler look-up table
        grid_doppler : isce3.core.LUT2d
            The grid doppler look-up table
        threshold_geo2rdr : float
            Convergence threashold for the geo2rdr
        numiter_geo2rdr : float
            Maximum number of iteration for geo2rdr
        delta_range : float
            An increment of the slant range to calculate the doppler rate
        """
        cube_group = self.require_group(sec_cube_group_path)
        cube_shape = [len(heights), geogrid.length, geogrid.width]

        zds, yds, xds = set_get_geo_info(self, sec_cube_group_path,
                                         geogrid,
                                         z_vect=heights,
                                         flag_cube=True)
        # seconds since ref epoch
        ref_epoch = radar_grid.ref_epoch
        ref_epoch_str = ref_epoch.isoformat()
        az_coord_units = f'seconds since {ref_epoch_str}'

        slant_range_raster = _get_raster_from_hdf5_ds(
            cube_group, 'secondarySlantRange', np.float64, cube_shape,
            zds=zds, yds=yds, xds=xds,
            long_name='slant-range',
            descr='Slant range of the secondary RSLC in meters',
            units='meters')
        azimuth_time_raster = _get_raster_from_hdf5_ds(
            cube_group, 'secondaryZeroDopplerAzimuthTime', np.float64, cube_shape,
            zds=zds, yds=yds, xds=xds,
            long_name='zero-Doppler azimuth time',
            descr='Zero doppler azimuth time of the secondary RSLC image',
            units=az_coord_units)

        isce3.geometry.make_radar_grid_cubes(radar_grid, geogrid, heights,
                                             orbit, native_doppler,
                                             grid_doppler,
                                             slant_range_raster,
                                             azimuth_time_raster,
                                             None, None, None, None,
                                             None, None, None,
                                             threshold_geo2rdr,
                                             numiter_geo2rdr,
                                             delta_range,
                                             flag_set_output_rasters_geolocation=False)

    def add_radar_grid_cubes(self):
        """
        Add the radar grid cubes
        """
        proc_cfg = self.cfg["processing"]
        radar_grid_cubes_geogrid = proc_cfg["radar_grid_cubes"]["geogrid"]
        radar_grid_cubes_heights = proc_cfg["radar_grid_cubes"]["heights"]

        threshold_geo2rdr = proc_cfg["geo2rdr"]["threshold"]
        iteration_geo2rdr = proc_cfg["geo2rdr"]["maxiter"]

        # Retrieve the group
        radar_grid_path = self.group_paths.RadarGridPath
        radar_grid = self.require_group(radar_grid_path)

        # Pull the doppler information
        cube_freq = "A" if "A" in self.freq_pols else "B"
        cube_rdr_grid = self.ref_rslc.getRadarGrid(cube_freq)
        cube_native_doppler = self.ref_rslc.getDopplerCentroid(
            frequency=cube_freq
        )
        cube_native_doppler.bounds_error = False
        grid_zero_doppler = LUT2d()

        add_radar_grid_cubes_to_hdf5(
            self,
            radar_grid_path,
            radar_grid_cubes_geogrid,
            radar_grid_cubes_heights,
            cube_rdr_grid,
            self.ref_orbit,
            cube_native_doppler,
            grid_zero_doppler,
            threshold_geo2rdr,
            iteration_geo2rdr,
            )

        # Update the radar grids attributes
        radar_grid['slantRange'].attrs['description'] = \
            np.string_("Slant range in meters")
        radar_grid['slantRange'].attrs['units'] = Units.meter

        zero_dopp_azimuth_time_units = \
            radar_grid['zeroDopplerAzimuthTime'].attrs['units']
        time_str = extract_datetime_from_string(
            str(zero_dopp_azimuth_time_units),
            'seconds since ')
        if time_str is not None:
            zero_dopp_azimuth_time_units = time_str
        radar_grid['zeroDopplerAzimuthTime'].attrs['units'] = \
            np.string_(zero_dopp_azimuth_time_units)

        radar_grid['projection'][...] = \
            radar_grid['projection'][()].astype(np.uint32)

        radar_grid['heightAboveEllipsoid'][...] = \
            radar_grid['heightAboveEllipsoid'][()].astype(np.float64)
        radar_grid['heightAboveEllipsoid'].attrs['description'] = \
            np.string_("Height values above WGS84 Ellipsoid"
                       " corresponding to the radar grid")
        radar_grid['heightAboveEllipsoid'].attrs['units'] = \
            Units.meter

        radar_grid['xCoordinates'].attrs['description'] = \
            np.string_("X coordinates corresponding to the radar grid")
        radar_grid['xCoordinates'].attrs['long_name'] = \
            np.string_("X coordinates of projection")
        radar_grid['yCoordinates'].attrs['description'] = \
            np.string_("Y coordinates corresponding to the radar grid")
        radar_grid['yCoordinates'].attrs['long_name'] = \
            np.string_("Y coordinates of projection")

        radar_grid['incidenceAngle'].attrs['description'] = \
            np.string_("Incidence angle is defined as the angle"
                       " between the LOS vector and the normal to"
                       " the ellipsoid at the target height")
        radar_grid['incidenceAngle'].attrs['long_name'] = \
            np.string_("Incidence angle")

        # Add the min and max attributes to the following dataset
        ds_names = ["incidenceAngle",
                    "losUnitVectorX",
                    "losUnitVectorY",
                    "alongTrackUnitVectorX",
                    "alongTrackUnitVectorY",
                    "elevationAngle"]

        for ds_name in ds_names:
            ds = radar_grid[ds_name][()]
            valid_min, valid_max = np.nanmin(ds), np.nanmax(ds)
            radar_grid[ds_name].attrs["min"] = valid_min
            radar_grid[ds_name].attrs["max"] = valid_max

            if ds_name not in ["incidenceAngle", "elevationAngle"]:
                radar_grid[ds_name].attrs["units"] = \
                    Units.unitless
        radar_grid["elevationAngle"].attrs["description"] = \
            np.string_("Elevation angle is defined as the angle between"
                       " the LOS vector and the normal to"
                       " the ellipsoid at thesensor")
        radar_grid["groundTrackVelocity"].attrs["description"] = \
            np.string_("Absolute value of the platform velocity"
                       " scaled at the target height")
        radar_grid["groundTrackVelocity"].attrs["units"] = \
            np.string_("meters / second")

        # Add the baseline dataset to radargrid
        self.add_baseline_info_to_cubes(radar_grid,
                                        radar_grid_cubes_geogrid,
                                        is_geogrid = True)

        # Add the secondary slant range and azimuth time cubes to the radarGrid cubes
        sec_cube_rdr_grid = self.sec_rslc.getRadarGrid(cube_freq)
        sec_cube_native_doppler = self.sec_rslc.getDopplerCentroid(
            frequency=cube_freq
        )

        self.add_secondary_radar_grid_cube(radar_grid_path,
                                            radar_grid_cubes_geogrid,
                                            radar_grid_cubes_heights,
                                            sec_cube_rdr_grid,
                                            self.sec_orbit, sec_cube_native_doppler,
                                            grid_zero_doppler,
                                            threshold_geo2rdr,
                                            iteration_geo2rdr,
                                            )

    def add_geocoding_to_algo_group(self):
        """
        Add the geocoding  group to algorithms group
        """
        pcfg_geocode = self.cfg["processing"]["geocode"]
        complex_interpolation = pcfg_geocode["wrapped_interferogram"][
            "interp_method"
        ]

        dem_interpolation = "biquintic"
        floating_interpolation = pcfg_geocode["interp_method"]
        integer_interpolation = "nearest"

        ds_params = [
            DatasetParams(
                "complexGeocodingInterpolation",
                complex_interpolation,
                "Geocoding interpolation algorithm for complex-valued"
                " datasets",
                {
                    "algorithm_type": "Geocoding",
                },
            ),
            DatasetParams(
                "demInterpolation",
                dem_interpolation,
                "DEM interpolation algorithm",
                {
                    "algorithm_type": "Geocoding",
                },
            ),
            DatasetParams(
                "floatingGeocodingInterpolation",
                floating_interpolation,
                "Geocoding interpolation algorithm for floating point"
                " datasets",
                {
                    "algorithm_type": "Geocoding",
                },
            ),
            DatasetParams(
                "integerGeocodingInterpolation",
                integer_interpolation,
                "Geocoding interpolation algorithm for integer datasets",
                {
                    "algorithm_type": "Geocoding",
                },
            ),
        ]

        geocoding_group = \
            self.require_group(f"{self.group_paths.AlgorithmsPath}/geocoding")
        for ds_param in ds_params:
            add_dataset_and_attrs(geocoding_group, ds_param)

    def add_geocoding_to_procinfo_params_group(self):
        """
        Add the geocoding  group to processingInformation/parameters group
        """
        proc_pcfg = self.cfg["processing"]
        iono = proc_pcfg["ionosphere_phase_correction"]["enabled"]
        wet_tropo = proc_pcfg["troposphere_delay"]["enable_wet_product"]
        dry_tropo = proc_pcfg["troposphere_delay"]["enable_hydrostatic_product"]

        # if the troposphere delay is not enabled
        if not proc_pcfg["troposphere_delay"]["enabled"]:
            wet_tropo = False
            dry_tropo = False

        ds_params = [
            DatasetParams(
                "azimuthIonosphericCorrectionApplied",
                np.string_(str(iono)),
                "Flag to indicate if the azimuth ionospheric correction is"
                " applied to improve geolocation"
                ,
            ),
            DatasetParams(
                "rangeIonosphericCorrectionApplied",
                np.string_(str(iono)),
                "Flag to indicate if the range ionospheric correction is"
                " applied to improve geolocation"
                ,
            ),
            DatasetParams(
                "wetTroposphericCorrectionApplied",
                np.string_(str(wet_tropo)),
                "Flag to indicate if the wet tropospheric correction is"
                " applied to improve geolocation"
                ,
            ),
            DatasetParams(
                "hydrostaticTroposphericCorrectionApplied",
                np.string_(str(dry_tropo)),
                "Flag to indicate if the hydrostatic tropospheric correction is"
                " applied to improve geolocation"
                ,
            ),
        ]

        group = self.require_group(
            f"{self.group_paths.ParametersPath}/geocoding"
        )
        for ds_param in ds_params:
            add_dataset_and_attrs(group, ds_param)

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithms to processingInformation group
        """
        super().add_algorithms_to_procinfo_group()
        self.add_geocoding_to_algo_group()

    def add_parameters_to_procinfo_group(self):
        """
        Add parameters group to processingInformation/parameters group
        """
        super().add_parameters_to_procinfo_group()
        self.add_geocoding_to_procinfo_params_group()

    def add_grids_to_hdf5(self):
        """
        Add grids to HDF5
        """
        # only add the common fields such as listofpolarizations, pixeloffset, and centerfrequency
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            grids_freq_group_name = (
                f"{self.group_paths.GridsPath}/frequency{freq}"
            )
            grids_freq_group = self.require_group(grids_freq_group_name)

            # Create the pixeloffsets group
            offset_group_name = f"{grids_freq_group_name}/pixelOffsets"
            self.require_group(offset_group_name)

            # center frequency and sub swaths groups of the RSLC
            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"
            ]

            list_of_pols = DatasetParams(
                "listOfPolarizations",
                np.string_(pol_list),
                "List of processed polarization layers with"
                f" frequency {freq}"
                ,
            )
            add_dataset_and_attrs(grids_freq_group, list_of_pols)

            rslc_freq_group.copy(
                "processedCenterFrequency",
                grids_freq_group,
                "centerFrequency",
            )

            # Add the description and units
            cfreq = grids_freq_group["centerFrequency"]
            cfreq.attrs['description'] = np.string_(" Center frequency of"
                                                    " the processed image in hertz")
            cfreq.attrs['units'] = Units.hertz
