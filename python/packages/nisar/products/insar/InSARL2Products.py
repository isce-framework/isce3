import h5py
import numpy as np
from isce3.core import LUT2d
from isce3.product import GeoGridParameters
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows.h5_prep import add_radar_grid_cubes_to_hdf5
from nisar.workflows.helpers import get_cfg_freq_pols

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSARBase import InSARWriter
from .InSARL1Products import L1InSARWriter
from .product_paths import L2GroupsPaths


class L2InSARWriter(L1InSARWriter):
    """
    Writer class for L2InSARWriter product inherent from L1InSARWriter (e.g. GOFF and GUNW)
    """

    def __init__(
        self,
        **kwds,
    ):
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

        InSARWriter.save_to_hdf5(self)

        self.add_radar_grid_cubes()
        self.add_grids_to_hdf5()

    def add_radar_grid_cubes(self):
        """
        Add the radar grid cubes
        """

        orbit_file = self.cfg["dynamic_ancillary_file_group"]["orbit"].get(
            "reference_orbit_file"
        )

        pcfg = self.cfg["processing"]
        radar_grid_cubes_geogrid = pcfg["radar_grid_cubes"]["geogrid"]
        radar_grid_cubes_heights = pcfg["radar_grid_cubes"]["heights"]

        threshold_geo2rdr = pcfg["geo2rdr"]["threshold"]
        iteration_geo2rdr = pcfg["geo2rdr"]["maxiter"]

        # Retrieve the group
        radarg_grid_path = self.group_paths.RadarGridPath
        self.require_group(radarg_grid_path)

        # cube geogrid
        cube_geogrid = GeoGridParameters(
            start_x=radar_grid_cubes_geogrid.start_x,
            start_y=radar_grid_cubes_geogrid.start_y,
            spacing_x=radar_grid_cubes_geogrid.spacing_x,
            spacing_y=radar_grid_cubes_geogrid.spacing_y,
            width=int(radar_grid_cubes_geogrid.width),
            length=int(radar_grid_cubes_geogrid.length),
            epsg=radar_grid_cubes_geogrid.epsg,
        )

        # Pull the orbit object
        if orbit_file is not None:
            orbit = load_orbit_from_xml(orbit_file)
        else:
            orbit = self.ref_rslc.getOrbit()

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
            radarg_grid_path,
            cube_geogrid,
            radar_grid_cubes_heights,
            cube_rdr_grid,
            orbit,
            cube_native_doppler,
            grid_zero_doppler,
            threshold_geo2rdr,
            iteration_geo2rdr,
        )

    def add_geocoding_to_algo(self, algo_group: h5py.Group):
        """
        Add the geocoding  group to algorithms group

        Parameters
        ------
        - algo_group(h5py.Group): algorithms group object
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
                np.string_(complex_interpolation),
                np.string_(
                    "Geocoding interpolation algorithm for complex-valued"
                    " datasets"
                ),
                {
                    "algorithm_type": np.string_("Geocoding"),
                },
            ),
            DatasetParams(
                "demInterpolation",
                np.string_(dem_interpolation),
                np.string_("DEM interpolation algorithm"),
                {
                    "algorithm_type": np.string_("Geocoding"),
                },
            ),
            DatasetParams(
                "floatingGeocodingInterpolation",
                np.string_(floating_interpolation),
                np.string_(
                    "Geocoding interpolation algorithm for floating point"
                    " datasets"
                ),
                {
                    "algorithm_type": np.string_("Geocoding"),
                },
            ),
            DatasetParams(
                "integerGeocodingInterpolation",
                np.string_(integer_interpolation),
                np.string_(
                    "Geocoding interpolation algorithm for integer datasets"
                ),
                {
                    "algorithm_type": np.string_("Geocoding"),
                },
            ),
        ]

        geocoding_group = algo_group.require_group("geocoding")
        for ds_param in ds_params:
            add_dataset_and_attrs(geocoding_group, ds_param)

    def add_geocoding_to_procinfo_params(self):
        """
        Add the geocoding  group to processingInformation/parameters group
        """

        pcfg = self.cfg["processing"]
        iono = pcfg["ionosphere_phase_correction"]["enabled"]
        wet_tropo = pcfg["troposphere_delay"]["enable_wet_product"]
        dry_tropo = pcfg["troposphere_delay"]["enable_dry_product"]

        # if the troposphere delay is not enabled
        if not pcfg["troposphere_delay"]["enabled"]:
            wet_tropo = False
            dry_tropo = False

        ds_params = [
            DatasetParams(
                "azimuthIonosphericCorrectionApplied",
                np.bool_(iono),
                np.string_(
                    "Flag to indicate if the azimuth ionospheric correction is"
                    " applied to improve geolocation"
                ),
            ),
            DatasetParams(
                "rangeIonosphericCorrectionApplied",
                np.bool_(iono),
                np.string_(
                    "Flag to indicate if the range ionospheric correction is"
                    " applied to improve geolocation"
                ),
            ),
            DatasetParams(
                "wetTroposphericCorrectionApplied",
                np.bool_(wet_tropo),
                np.string_(
                    "Flag to indicate if the wet tropospheric correction is"
                    " applied to improve geolocation"
                ),
            ),
            DatasetParams(
                "dryTroposphericCorrectionApplied",
                np.bool_(dry_tropo),
                np.string_(
                    "Flag to indicate if the dry tropospheric correction is"
                    " applied to improve geolocation"
                ),
            ),
        ]

        group = self.require_group(
            f"{self.group_paths.ParametersPath}/geocoding"
        )
        for ds_param in ds_params:
            add_dataset_and_attrs(group, ds_param)

    def add_algorithms_to_procinfo(self):
        """
        Add the algorithms to processingInformation group

        Return
        ------
        algo_group (h5py.Group): the algorithm group object
        """

        algo_group = super().add_algorithms_to_procinfo()
        self.add_geocoding_to_algo(algo_group)

        return algo_group

    def add_parameters_to_procinfo(self):
        """
        Add parameters group to processingInformation/parameters group
        """

        super().add_parameters_to_procinfo()
        self.add_geocoding_to_procinfo_params()

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
                np.string_(
                    "List of processed polarization layers with"
                    f" frequency{freq}"
                ),
            )
            add_dataset_and_attrs(grids_freq_group, list_of_pols)

            self._copy_dataset_by_name(
                rslc_freq_group,
                "processedCenterFrequency",
                grids_freq_group,
                "centerFrequency",
            )
