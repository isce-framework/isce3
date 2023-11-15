import numpy as np
from isce3.core import LUT2d
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows.h5_prep import add_radar_grid_cubes_to_hdf5
from nisar.workflows.helpers import get_cfg_freq_pols

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_base_writer import InSARBaseWriter
from .InSAR_L1_writer import L1InSARWriter
from .product_paths import L2GroupsPaths


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

    def add_radar_grid_cubes(self):
        """
        Add the radar grid cubes
        """
        orbit_file = self.cfg["dynamic_ancillary_file_group"]\
            ["orbit_files"].get("reference_orbit_file")

        proc_cfg = self.cfg["processing"]
        radar_grid_cubes_geogrid = proc_cfg["radar_grid_cubes"]["geogrid"]
        radar_grid_cubes_heights = proc_cfg["radar_grid_cubes"]["heights"]

        threshold_geo2rdr = proc_cfg["geo2rdr"]["threshold"]
        iteration_geo2rdr = proc_cfg["geo2rdr"]["maxiter"]

        # Retrieve the group
        radarg_grid_path = self.group_paths.RadarGridPath
        self.require_group(radarg_grid_path)

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
            radar_grid_cubes_geogrid,
            radar_grid_cubes_heights,
            cube_rdr_grid,
            orbit,
            cube_native_doppler,
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
                f" frequency{freq}"
                ,
            )
            add_dataset_and_attrs(grids_freq_group, list_of_pols)

            rslc_freq_group.copy(
                "processedCenterFrequency",
                grids_freq_group,
                "centerFrequency",
            )
