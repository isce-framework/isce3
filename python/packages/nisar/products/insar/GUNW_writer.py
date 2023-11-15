import h5py
import numpy as np
from nisar.workflows.h5_prep import set_get_geo_info
from nisar.workflows.helpers import get_cfg_freq_pols

from .InSAR_base_writer import InSARBaseWriter
from .InSAR_L2_writer import L2InSARWriter
from .InSAR_products_info import InSARProductsInfo
from .product_paths import GUNWGroupsPaths
from .RIFG_writer import RIFGWriter
from .RUNW_writer import RUNWWriter


class GUNWWriter(RUNWWriter, RIFGWriter, L2InSARWriter):
    """
    Writer class for GUNW product inherent from both the RUNWWriter,
    RIFGWriter, and the L2InSARWriter
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
        self.attrs["reference_document"] = \
            np.string_("D-102272 NISAR NASA SDS Product Specification"
                       " L2 Geocoded Unwrapped Interferogram")

        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(self["/"].id, np.string_("complex64"))

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithms to processingInformation group
        """
        RUNWWriter.add_algorithms_to_procinfo_group(self)
        L2InSARWriter.add_geocoding_to_algo_group(self)

    def add_parameters_to_procinfo_group(self):
        """
        Add parameters group to processingInformation/parameters group
        """
        RUNWWriter.add_parameters_to_procinfo_group(self)

        # the unwrappedInterfergram group under the processingInformation/parameters
        # group is copied from the RUNW product, but the name in RUNW product is
        # 'interferogram', while in GUNW its name is 'unwrappedInterferogram'. Here
        # is to rename the interfegram group name to unwrappedInterferogram group name
        old_igram_group_name = \
            f"{self.group_paths.ParametersPath}/interferogram"
        new_igram_group_name = \
            f"{self.group_paths.ParametersPath}/unwrappedInterferogram"
        self.move(old_igram_group_name, new_igram_group_name)

        # the wrappedInterfergram group under the processingInformation/parameters
        # group is copied from the RIFG product, but the name in RIFG product is
        # 'interferogram', while in GUNW its name is 'wrappedInterferogram'. Here
        # is to rename the interfegram group name to wrappedInterferogram group name
        RIFGWriter.add_interferogram_to_procinfo_params_group(self)
        new_igram_group_name = \
            f"{self.group_paths.ParametersPath}/wrappedInterferogram"
        self.move(old_igram_group_name, new_igram_group_name)

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

        # Only add the common fields such as list of polarizations, pixel offsets, and center frequency
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            grids_freq_group_name = (
                f"{self.group_paths.GridsPath}/frequency{freq}"
            )
            grids_freq_group = self.require_group(grids_freq_group_name)

            # Create the pixeloffsets group
            offset_group_name = f"{grids_freq_group_name}/pixelOffsets"
            self.require_group(offset_group_name)

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

            # shape of the wrapped interferogram
            wrapped_shape = (
                wrapped_geogrids.length,
                wrapped_geogrids.width,
            )

            unwrapped_group_name = \
                f"{grids_freq_group_name}/unwrappedInterferogram"

            wrapped_group_name = \
                f"{grids_freq_group_name}/wrappedInterferogram"

            pixeloffsets_group_name = \
                f"{grids_freq_group_name}/pixelOffsets"

            unwrapped_group = self.require_group(unwrapped_group_name)

            # set the geo information for the mask
            yds, xds = set_get_geo_info(
                self,
                unwrapped_group_name,
                unwrapped_geogrids,
            )

            # Prepare 2d mask dataset
            self._create_2d_dataset(
                unwrapped_group,
                "mask",
                unwrapped_shape,
                np.uint8,
                "Byte layer with flags for various channels"
                " (e.g. layover/shadow, data quality)"
                ,
                "DN",
                grids_val,
                xds=xds,
                yds=yds)

            for pol in pol_list:
                unwrapped_pol_name = f"{unwrapped_group_name}/{pol}"
                unwrapped_pol_group = self.require_group(unwrapped_pol_name)

                yds, xds = set_get_geo_info(
                    self,
                    unwrapped_pol_name,
                    unwrapped_geogrids,
                )

                #unwrapped dataset parameters as tuples in the following
                #order: dataset name, data type, description, and units
                unwrapped_ds_params = [
                    ("coherenceMagnitude", np.float32,
                     f"Coherence magnitude between {pol} layers",
                     "unitless"),
                    ("connectedComponents", np.uint32,
                     f"Connected components for {pol} layers",
                     "DN"),
                    ("ionospherePhaseScreen", np.float32,
                     "Ionosphere phase screen",
                     "radians"),
                    ("ionospherePhaseScreenUncertainty", np.float32,
                     "Uncertainty of the ionosphere phase screen",
                     "radians"),
                    ("unwrappedPhase", np.float32,
                    f"Unwrapped interferogram between {pol} layers",
                     "radians"),
                ]

                for ds_param in unwrapped_ds_params:
                    ds_name, ds_datatype, ds_description, ds_unit\
                        = ds_param
                    self._create_2d_dataset(
                        unwrapped_pol_group,
                        ds_name,
                        unwrapped_shape,
                        ds_datatype,
                        ds_description,
                        ds_unit,
                        grids_val,
                        xds=xds,
                        yds=yds)

                wrapped_pol_name = f"{wrapped_group_name}/{pol}"
                wrapped_pol_group = self.require_group(wrapped_pol_name)

                yds, xds = set_get_geo_info(
                    self,
                    wrapped_pol_name,
                    wrapped_geogrids,
                )

                #wrapped dataset parameters as tuples in the following
                #order: the dataset name,data type, description, and units
                wrapped_ds_params = [
                    ("coherenceMagnitude", np.float32,
                     f"Coherence magnitude between {pol} layers",
                     "unitless"),
                    ("wrappedInterferogram", np.complex64,
                     f"Complex wrapped interferogram between {pol} layers",
                     "DN"),
                ]

                for ds_param in wrapped_ds_params:
                    ds_name, ds_datatype, ds_description, ds_unit\
                        = ds_param
                    self._create_2d_dataset(
                        wrapped_pol_group,
                        ds_name,
                        wrapped_shape,
                        ds_datatype,
                        ds_description,
                        ds_unit,
                        grids_val,
                        xds=xds,
                        yds=yds,
                    )

                pixeloffsets_pol_name = f"{pixeloffsets_group_name}/{pol}"
                pixeloffsets_pol_group = self.require_group(
                    pixeloffsets_pol_name
                )

                yds, xds = set_get_geo_info(
                    self,
                    pixeloffsets_pol_name,
                    unwrapped_geogrids,
                )

                # pixel offsets dataset parameters as tuples in the following
                # order: dataset name,data type, description, and units
                pixel_offsets_ds_params = [
                    ("alongTrackOffset", np.float32,
                     "Along track offset",
                     "meters"),
                    ("correlationSurfacePeak", np.float32,
                     "Normalized cross-correlation surface peak",
                     "unitless"),
                    ("slantRangeOffset", np.float32,
                     "Slant range offset",
                     "meters"),
                ]

                for ds_param in pixel_offsets_ds_params:
                    ds_name, ds_datatype, ds_description, ds_unit\
                        = ds_param
                    self._create_2d_dataset(
                        pixeloffsets_pol_group,
                        ds_name,
                        unwrapped_shape,
                        ds_datatype,
                        ds_description,
                        ds_unit,
                        grids_val,
                        xds=xds,
                        yds=yds,
                    )