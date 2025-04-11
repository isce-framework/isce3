import numpy as np
from nisar.workflows.h5_prep import set_get_geo_info
from nisar.workflows.helpers import get_cfg_freq_pols

from .InSAR_base_writer import InSARBaseWriter
from .InSAR_HDF5_optimizer_config import get_InSAR_output_options
from .InSAR_L2_writer import L2InSARWriter
from .InSAR_products_info import InSARProductsInfo
from .product_paths import GOFFGroupsPaths
from .ROFF_writer import ROFFWriter
from .units import Units

class GOFFWriter(ROFFWriter, L2InSARWriter):
    """
    Writer class for GOFF product inherent from both
    ROFFWriter and L2InSARWriter
    """

    def __init__(self, **kwds):
        """
        Constructor for GOFF class
        """
        hdf5_opt_config, kwds = get_InSAR_output_options(kwds, 'GOFF')

        super().__init__(**kwds)

        # HDF5 IO optimizer configuration
        self.hdf5_optimizer_config = hdf5_opt_config

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
            np.bytes_("D-105010 NISAR NASA SDS Product Specification"
                       " L2 Geocoded Pixel Offsets")

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithms to processingInformation group
        """
        ROFFWriter.add_algorithms_to_procinfo_group(self)
        L2InSARWriter.add_geocoding_to_algo_group(self)

    def add_parameters_to_procinfo_group(self):
        """
        Add parameters group to
        processingInformation/parameters group
        """
        ROFFWriter.add_parameters_to_procinfo_group(self)
        L2InSARWriter.add_geocoding_to_procinfo_params_group(self)

        # Update the descriptions of the reference and secondary
        for rslc_name in ['reference', 'secondary']:
            rslc = self[self.group_paths.ParametersPath][rslc_name]
            rslc['referenceTerrainHeight'].attrs['description'] = \
                np.bytes_("Reference Terrain Height as a function of"
                           f" map coordinates for {rslc_name} RSLC")
            rslc['referenceTerrainHeight'].attrs['units'] = \
                Units.meter


    def add_grids_to_hdf5(self):
        """
        Add grids to HDF5
        """
        L2InSARWriter.add_grids_to_hdf5(self)

        proc_cfg = self.cfg["processing"]
        geogrids = proc_cfg["geocode"]["geogrids"]
        grids_val = np.bytes_("projection")

        # Extract offset layer names for later processing
        layers = [
            layer
            for layer in proc_cfg["offsets_product"]
            if layer.startswith("layer")]

        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # add the common fields such as listofpolarizations, pixeloffset,
            # and centerfrequency
            grids_freq_group_name = \
                f"{self.group_paths.GridsPath}/frequency{freq}"

            grids_freq_group = self.require_group(grids_freq_group_name)

            offset_group_name = f"{grids_freq_group_name}/pixelOffsets"
            offset_group = self.require_group(offset_group_name)

            goff_geogrids = geogrids[freq]
            goff_shape = (goff_geogrids.length,goff_geogrids.width)

            # set the geo information for the mask
            yds, xds = set_get_geo_info(
                self,
                offset_group_name,
                goff_geogrids,
            )

            self._create_2d_dataset(
                offset_group,
                "mask",
                goff_shape,
                np.uint8,
                ("Combination of water mask and a mask of subswaths of valid samples"
                 " in the reference RSLC and geometrically-coregistered secondary RSLC."
                 " Each pixel value is a three-digit number:"
                 " the most significant digit represents the water flag of that pixel in the reference RSLC,"
                 " where 1 is water and 0 is non-water;"
                 " the second digit represents the subswath number of that pixel in the reference RSLC;"
                 " the least-significant digit represents the subswath number of that pixel in the secondary RSLC."
                 " A value of '0' in either subswath digit indicates an invalid sample in the corresponding RSLC"),
                grid_mapping=grids_val,
                xds=xds,
                yds=yds,
                fill_value=255,
            )
            offset_group['mask'].attrs['valid_min'] = 0
            offset_group['mask'].attrs['percentage_water'] = 0.0

            pixeloffsets_group_name = \
                f"{grids_freq_group_name}/pixelOffsets"

            # add the list of layers
            self.add_list_of_layers(grids_freq_group)

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

                    pixeloffsets_pol_layer_group['projection'][...] = \
                        pixeloffsets_pol_layer_group['projection'][()].astype(np.uint32)
                    pixeloffsets_pol_layer_group['yCoordinateSpacing'].attrs['long_name'] = \
                        np.bytes_("Y coordinates spacing")
                    pixeloffsets_pol_layer_group['xCoordinateSpacing'].attrs['long_name'] = \
                        np.bytes_("X coordinates spacing")
                    pixeloffsets_pol_layer_group['xCoordinates'].attrs['long_name'] = \
                        np.bytes_("X coordinates of projection")
                    pixeloffsets_pol_layer_group['yCoordinates'].attrs['long_name'] = \
                        np.bytes_("Y coordinates of projection")

                    #pixeloffsets dataset parameters as tuples in the following
                    #order: dataset name, description, and units
                    pixeloffsets_ds_params = [
                        ("alongTrackOffset", np.float32,
                         "Raw (unculled, unfiltered) along-track pixel offsets",
                         Units.meter),
                        ("slantRangeOffset",  np.float32,
                         "Raw (unculled, unfiltered) slant range pixel offsets",
                         Units.meter),
                        ("alongTrackOffsetVariance", np.float32,
                         "Along-track pixel offsets variance",
                         Units.meter2),
                        ("slantRangeOffsetVariance", np.float32,
                         "Slant range pixel offsets variance",
                         Units.meter2),
                        ("crossOffsetVariance", np.float32,
                         "Off-diagonal term of the pixel offsets covariance matrix",
                         Units.meter2),
                        ("correlationSurfacePeak", np.float32,
                         "Normalized correlation surface peak",
                         Units.unitless),
                        ("snr", np.float32,
                         "Pixel offsets signal-to-noise ratio",
                         Units.unitless),
                    ]

                    for ds_params in pixeloffsets_ds_params:
                        ds_name, ds_type, ds_description, ds_units = ds_params
                        self._create_2d_dataset(
                            pixeloffsets_pol_layer_group,
                            ds_name,
                            goff_shape,
                            ds_type,
                            ds_description,
                            ds_units,
                            grids_val,
                            xds=xds,
                            yds=yds,
                        )
