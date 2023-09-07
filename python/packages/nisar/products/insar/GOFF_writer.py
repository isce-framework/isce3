import numpy as np
from nisar.workflows.h5_prep import set_get_geo_info
from nisar.workflows.helpers import get_cfg_freq_pols

from .InSAR_products_info import InSARProductsInfo
from .InSAR_base_writer import InSARBaseWriter
from .InSAR_L2_writer import L2InSARWriter
from .product_paths import GOFFGroupsPaths
from .ROFF_writer import ROFFWriter


class GOFFWriter(ROFFWriter, L2InSARWriter):
    """
    Writer class for GOFF product inherent from both
    ROFFWriter and L2InSARWriter
    """

    def __init__(self, **kwds):
        """
        Constructor for GOFF class
        """

        super().__init__(**kwds)

        # group paths are GOFF group paths
        self.group_paths = GOFFGroupsPaths()

        # GOFF product information
        self.product_info = InSARProductsInfo.GOFF()

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

        self.attrs["title"] = "NISAR L2 GOFF Product"
        self.attrs["reference_document"] = \
            np.string_("D-105010 NISAR NASA SDS Product Specification"
                       " L2 Geocoded Pixel Offsets")

    def add_algorithms_to_procinfo(self):
        """
        Add the algorithms to processingInformation group
        """
        ROFFWriter.add_algorithms_to_procinfo_group(self)
        L2InSARWriter.add_geocoding_to_algo_group(self)

    def add_parameters_to_procinfo(self):
        """
        Add parameters group to processingInformation/parameters group
        """
        ROFFWriter.add_parameters_to_procinfo_group(self)
        L2InSARWriter.add_geocoding_to_procinfo_params_group(self)

    def add_grids_to_hdf5(self):
        """
        Add grids to HDF5
        """
        L2InSARWriter.add_grids_to_hdf5(self)

        proc_cfg = self.cfg["processing"]
        geogrids = proc_cfg["geocode"]["geogrids"]
        grids_val = np.string_("projection")
        layers = [
            layer
            for layer in proc_cfg["offsets_product"]
            if layer.startswith("layer")
        ]

        # only add the common fields such as listofpolarizations, pixeloffset,
        # and centerfrequency
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            grids_freq_group_name = (
                f"{self.group_paths.GridsPath}/frequency{freq}"
            )
            grids_freq_group = self.require_group(grids_freq_group_name)

            offset_group_name = f"{grids_freq_group_name}/pixelOffsets"
            self.require_group(offset_group_name)

            goff_geogrids = geogrids[freq]
            goff_shape = (
                goff_geogrids.length,
                goff_geogrids.width,
            )
            pixeloffsets_group_name = \
                f"{grids_freq_group_name}/pixelOffsets"

            for pol in pol_list:
                for layer in layers:
                    pixeloffsets_pol_layer_name = \
                        f"{pixeloffsets_group_name}/{pol}/{layer}"
                    pixeloffsets_pol_layer_group = \
                        self.require_group(pixeloffsets_pol_layer_name)

                    yds, xds = set_get_geo_info(
                        self,
                        pixeloffsets_pol_layer_name,
                        goff_geogrids,
                    )

                    #pixeloffsets dataset parameters as tuples in the following
                    #order: dataset name, description, and units
                    pixeloffsets_ds_params = [
                        ("alongTrackOffset",
                         "Raw (unculled, unfiltered) along-track pixel offsets",
                         "meters"),
                        ("slantRangeOffset",
                         "Raw (unculled, unfiltered) slant range pixel offsets",
                         "meters"),
                        ("alongTrackOffsetVariance",
                         "Along-track pixel offsets variance",
                         "unitless"),
                        ("slantRangeOffsetVariance",
                         "Slant range pixel offsets variance",
                         "unitless"),
                        ("crossOffsetVariance",
                         "Off-diagonal term of the pixel offsets covariance matrix",
                         "unitless"),
                        ("correlationSurfacePeak",
                         "Normalized surface correlation peak",
                         "unitless"),
                        ("snr",
                         "Pixel offsets signal-to-noise ratio",
                         "unitless"),
                    ]

                    for ds_params in pixeloffsets_ds_params:
                        ds_name, ds_description, ds_units = ds_params
                        self._create_2d_dataset(
                            pixeloffsets_pol_layer_group,
                            ds_name,
                            goff_shape,
                            np.float32,
                            ds_description,
                            ds_units,
                            grids_val,
                            xds=xds,
                            yds=yds)