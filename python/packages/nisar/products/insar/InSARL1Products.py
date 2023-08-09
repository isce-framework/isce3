import numpy as np
from isce3.core import LUT2d
from isce3.product import RadarGridParameters
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows.h5_prep import add_geolocation_grid_cubes_to_hdf5

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSARBase import InSARWriter
from .product_paths import CommonPaths


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

    def add_identification_group(self):
        """
        Add identification group
        """
        super().add_identification_group()
        
        dst_id_group = self.require_group(CommonPaths.IdentificationPath)
        ds_params = [
            DatasetParams(
                "isGeocoded",
                np.bool_(False),
                "Flag to indicate radar geometry or geocoded product",
            ),
            DatasetParams(
                "productLevel",
                "L1",
                (
                    "Product level. L0A: Unprocessed instrument data; L0B:"
                    " Reformatted,unprocessed instrument data; L1: Processed"
                    " instrument data in radar coordinates system; and L2:"
                    " Processed instrument data in geocoded coordinates system"
                ),
            ),
        ]
        for ds_param in ds_params:
            add_dataset_and_attrs(dst_id_group, ds_param)

    def _get_geolocation_grid_cubes_path(self):
        """
        Get the geolocation grid cube path.
        To change the path for the children classes, need to overwrite this function
        
        """
        return ""

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

        # add geolocation grid cubes to hdf5
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

        # add the min and max attributes to the dataset
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