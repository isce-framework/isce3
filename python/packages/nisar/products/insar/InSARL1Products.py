import numpy as np
from isce3.core import LUT2d
from isce3.product import RadarGridParameters
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows.h5_prep import add_geolocation_grid_cubes_to_hdf5
from nisar.workflows.helpers import get_cfg_freq_pols

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSARBase import InSARWriter
from .product_paths import L1GroupsPaths


class L1InSARWriter(InSARWriter):
    """
    InSAR Level 1 prodcut writer (e.g. RIFG, RUNW, ROFF)
    """

    def __init__(
        self,
        **kwds,
    ):
        """
        Constructor for InSAR L1 product (RIFG, RUNW, and ROFF).
        """
        super().__init__(**kwds)

        # Level 1 product group path
        self.group_paths = L1GroupsPaths()
        
    def save_to_hdf5(self):
        """
        write to the HDF5
        """
        super().save_to_hdf5()
        
        self.add_geolocation_grid_cubes()
        self.add_swaths_to_hdf5()
    
    def _get_geolocation_grid_cubes_path(self):
        """
        Get the geolocation grid cube path.        
        """
        return self.group_paths.GeolocationGridPath

    def add_geolocation_grid_cubes(self):
        """
        Add the geolocation grid cubes
        """
        
        # Pull the orbit object
        if self.external_orbit_path is not None:
            orbit = load_orbit_from_xml(self.external_orbit_path)
        else:
            orbit = self.ref_rslc.getOrbit()

        # Retrieve the group
        geolocationGrid_path = self._get_geolocation_grid_cubes_path()
        self.require_group(geolocationGrid_path)

        # Pull the radar frequency
        cube_freq = "A" if "A" in self.freq_pols else "B"
        radargrid = RadarGridParameters(self.ref_h5_slc_file)

        # Default is [-500, 9000]
        heights = np.linspace(-500, 9000, 20)

        # Figure out decimation factors that give < 500 m spacing.
        max_spacing = 500.0
        t = (
            radargrid.sensing_mid
            + (radargrid.ref_epoch - orbit.reference_epoch).total_seconds()
        )
        _, v = orbit.interpolate(t)
        dx = np.linalg.norm(v) / radargrid.prf
        tskip = int(np.floor(max_spacing / dx))
        rskip = int(np.floor(max_spacing / radargrid.range_pixel_spacing))
        radargrid = radargrid[::tskip, ::rskip]

        grid_doppler = LUT2d()
        cube_native_doppler = self.ref_rslc.getDopplerCentroid(
            frequency=cube_freq
        )
        cube_native_doppler.bounds_error = False

        tol = dict(
            threshold_geo2rdr=1e-8,
            numiter_geo2rdr=50,
            delta_range=10,
        )

        # Add geolocation grid cubes to hdf5
        add_geolocation_grid_cubes_to_hdf5(
            self,
            geolocationGrid_path,
            radargrid,
            heights,
            orbit,
            cube_native_doppler,
            grid_doppler,
            4326,
            **tol,
        )

        # Add the min and max attributes to the dataset
        ds_names = [
            "incidenceAngle",
            "losUnitVectorX",
            "losUnitVectorY",
            "alongTrackUnitVectorX",
            "alongTrackUnitVectorY",
            "elevationAngle",
        ]
        geolocation_grid_group = self[geolocationGrid_path]
        for ds_name in ds_names:
            ds = geolocation_grid_group[ds_name][()]
            valid_min, valid_max = np.nanmin(ds), np.nanmax(ds)
            geolocation_grid_group[ds_name].attrs["min"] = valid_min
            geolocation_grid_group[ds_name].attrs["max"] = valid_max
            
            
    def add_algorithms_to_procinfo(self):
        """
        Add the algorithms group to the processingInformation group

        Return
        ------
        algo_group (h5py.Group): the algorithm group object
        """
        
        algo_group = super().add_algorithms_to_procinfo()
        self.add_coregistration_to_algo(algo_group)
        self.add_interferogramformation_to_algo(algo_group)
        
        return algo_group

    def add_parameters_to_procinfo(self):
        """
        Add the parameters group to the "processingInformation" group
        """
        
        super().add_parameters_to_procinfo()
        
        self.add_interferogram_to_procinfo_params()
        self.add_pixeloffsets_to_procinfo_params()


    def add_swaths_to_hdf5(self):
        """
        Add Swaths to the HDF5
        """ 
        
        # only add the common fields such as listofpolarizations, pixeloffset, and centerfrequency       
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            swaths_freq_group_name = (
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            )
            swaths_freq_group = self.require_group(swaths_freq_group_name)

            # Create the pixeloffsets group
            offset_group_name = f"{swaths_freq_group_name}/pixelOffsets"
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
            add_dataset_and_attrs(swaths_freq_group, list_of_pols)

            self._copy_dataset_by_name(
                rslc_freq_group,
                "processedCenterFrequency",
                swaths_freq_group,
                "centerFrequency",
            )