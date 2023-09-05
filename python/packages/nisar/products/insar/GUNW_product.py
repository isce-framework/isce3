import h5py
import numpy as np
from nisar.workflows.h5_prep import set_get_geo_info
from nisar.workflows.helpers import get_cfg_freq_pols

from .common import InSARProductsInfo
from .InSAR_base_writer import InSARBaseWriter
from .InSARL2Products import L2InSARWriter
from .product_paths import GUNWGroupsPaths
from .RUNW_writer import RUNWWriter


class GUNWWriter(RUNWWriter, L2InSARWriter):
    """
    Writer class for GUNW product inherent from RUNWWriter
    """
    def __init__(self, **kwds):
        """
        Constructor for GUNW writer class
        """
        super().__init__(**kwds)

        # group paths are GUNW group paths
        self.group_paths = GUNWGroupsPaths()

        # GUNW product information
        self.product_info = InSARProductsInfo.GUNW()

    def save_to_hdf5(self):
        """
        Save to HDF5
        """
        L2InSARWriter.save_to_hdf5(self)

    def add_root_attrs(self):
        """
        add root attributes
        """
        InSARBaseWriter.add_root_attrs(self)

        self.attrs["title"] = np.string_("NISAR L2 GUNW Product")
        self.attrs["reference_document"] = np.string_("JPL-102272")

        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(self["/"].id, np.string_("complex64"))

    def add_algorithms_to_procinfo(self):
        """
        Add the algorithms to processingInformation group
        """
        RUNWWriter.add_algorithms_to_procinfo_group(self)
        L2InSARWriter.add_geocoding_to_algo_group(self)

    def add_parameters_to_procinfo(self):
        """
        Add parameters group to processingInformation/parameters group
        """
        RUNWWriter.add_parameters_to_procinfo_group(self)
        L2InSARWriter.add_geocoding_to_procinfo_params_group(self)

    def add_grids_to_hdf5(self):
        """
        Add grids to HDF5
        """
        L2InSARWriter.add_grids_to_hdf5(self)

        pcfg = self.cfg["processing"]
        geogrids = pcfg["geocode"]["geogrids"]
        wrapped_igram_geogrids = pcfg["geocode"]["wrapped_igram_geogrids"]

        grids_val = np.string_("projection")

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

            rslc_freq_group.copy("numberOfSubSwaths",
                                 grids_freq_group)

            unwrapped_geogrids = geogrids[freq]
            wrapped_geogrids = wrapped_igram_geogrids[freq]

            # shape of the unwrapped phase
            unwrapped_shape = (
                unwrapped_geogrids.length,
                unwrapped_geogrids.width,
            )

            # shape of the wrapped interferogram shape
            wrapped_shape = (
                wrapped_geogrids.length,
                wrapped_geogrids.width,
            )

            unwrapped_group_name = (
                f"{grids_freq_group_name}/interferogram/unwrapped"
            )
            wrapped_group_name = (
                f"{grids_freq_group_name}/interferogram/wrapped"
            )
            pixeloffsets_group_name = (
                f"{grids_freq_group_name}/pixelOffsets"
            )

            unwrapped_group = self.require_group(unwrapped_group_name)

            # set the geo information for the mask
            yds, xds = set_get_geo_info(
                self,
                unwrapped_group_name,
                unwrapped_geogrids,
            )

            # Mask
            self._create_2d_dataset(
                unwrapped_group,
                "mask",
                unwrapped_shape,
                np.byte,
                np.string_(
                    "Byte layer with flags for various channels"
                    " (e.g.layover/shadow, data quality)"
                ),
                np.string_("DN"),
                grids_val,
                xds=xds,
                yds=yds,
            )

            for pol in pol_list:
                # unwrapped interferogram group
                unwrapped_pol_name = f"{unwrapped_group_name}/{pol}"
                unwrapped_pol_group = self.require_group(unwrapped_pol_name)

                yds, xds = set_get_geo_info(
                    self,
                    unwrapped_pol_name,
                    unwrapped_geogrids,
                )

                self._create_2d_dataset(
                    unwrapped_pol_group,
                    "coherenceMagnigtude",
                    unwrapped_shape,
                    np.float32,
                    np.string_(f"Coherence magnitude between {pol} layers"),
                    np.string_("unitless"),
                    grids_val,
                    xds=xds,
                    yds=yds,
                )
                self._create_2d_dataset(
                    unwrapped_pol_group,
                    "connectedComponents",
                    unwrapped_shape,
                    np.uint32,
                    np.string_(f"Connected components for {pol} layers"),
                    np.string_("DN"),
                    grids_val,
                    xds=xds,
                    yds=yds,
                    fill_value=0,
                )
                self._create_2d_dataset(
                    unwrapped_pol_group,
                    "ionospherePhaseScreen",
                    unwrapped_shape,
                    np.float32,
                    np.string_(f"Ionosphere phase screen"),
                    np.string_("radians"),
                    grids_val,
                    xds=xds,
                    yds=yds,
                )
                self._create_2d_dataset(
                    unwrapped_pol_group,
                    "ionospherePhaseScreenUncertainty",
                    unwrapped_shape,
                    np.float32,
                    np.string_(f"Uncertainty of the ionosphere phase screen"),
                    np.string_("radians"),
                    grids_val,
                    xds=xds,
                    yds=yds,
                )
                self._create_2d_dataset(
                    unwrapped_pol_group,
                    "unwrappedPhase",
                    unwrapped_shape,
                    np.float32,
                    np.string_(
                        f"Unwrapped interferogram between {pol} layers"
                    ),
                    np.string_("radians"),
                    grids_val,
                    xds=xds,
                    yds=yds,
                )

                # wrapped interferogram group
                wrapped_pol_name = f"{wrapped_group_name}/{pol}"
                wrapped_pol_group = self.require_group(wrapped_pol_name)

                yds, xds = set_get_geo_info(
                    self,
                    wrapped_pol_name,
                    wrapped_geogrids,
                )

                self._create_2d_dataset(
                    wrapped_pol_group,
                    "wrappedCoherenceMagnitude",
                    wrapped_shape,
                    np.float32,
                    np.string_(f"Coherence magnitude between {pol} layers"),
                    np.string_("unitless"),
                    grids_val,
                    xds=xds,
                    yds=yds,
                )
                self._create_2d_dataset(
                    wrapped_pol_group,
                    "wrappedInterferogram",
                    wrapped_shape,
                    np.complex64,
                    np.string_(
                        f"Complex wrapped interferogram between {pol} layers"
                    ),
                    np.string_("DN"),
                    grids_val,
                    xds=xds,
                    yds=yds,
                )

                # pixelOffsets group
                pixeloffsets_pol_name = f"{pixeloffsets_group_name}/{pol}"
                pixeloffsets_pol_group = self.require_group(
                    pixeloffsets_pol_name
                )

                yds, xds = set_get_geo_info(
                    self,
                    pixeloffsets_pol_name,
                    unwrapped_geogrids,
                )

                self._create_2d_dataset(
                    pixeloffsets_pol_group,
                    "alongTrackOffset",
                    unwrapped_shape,
                    np.float32,
                    np.string_(f"Along track offset"),
                    np.string_("meters"),
                    grids_val,
                    xds=xds,
                    yds=yds,
                )
                self._create_2d_dataset(
                    pixeloffsets_pol_group,
                    "crossCorrelationPeak",
                    unwrapped_shape,
                    np.float32,
                    np.string_(f"Normalized cross-correlation surface peak"),
                    np.string_("unitless"),
                    grids_val,
                    xds=xds,
                    yds=yds,
                )
                self._create_2d_dataset(
                    pixeloffsets_pol_group,
                    "slantRangeOffset",
                    unwrapped_shape,
                    np.float32,
                    np.string_(f"Slant range offset"),
                    np.string_("meters"),
                    grids_val,
                    xds=xds,
                    yds=yds,
                )
