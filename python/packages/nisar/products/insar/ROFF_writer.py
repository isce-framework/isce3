import numpy as np
from nisar.workflows.h5_prep import get_off_params
from nisar.workflows.helpers import get_cfg_freq_pols

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_base_writer import InSARBaseWriter
from .InSAR_HDF5_optimizer_config import get_InSAR_output_options
from .InSAR_L1_writer import L1InSARWriter
from .InSAR_products_info import InSARProductsInfo
from .product_paths import ROFFGroupsPaths
from .units import Units
from .utils import get_pixel_offsets_dataset_shape, get_pixel_offsets_params


class ROFFWriter(L1InSARWriter):
    """
    Writer class for ROFF product inherent from L1InSARWriter
    """
    def __init__(self, **kwds):
        """
        Constructor for ROFF class
        """
        hdf5_opt_config, kwds = get_InSAR_output_options(kwds, 'ROFF')

        super().__init__(**kwds)

        # HDF5 IO optimizer configuration
        self.hdf5_optimizer_config = hdf5_opt_config

        # group paths are ROFF group paths
        self.group_paths = ROFFGroupsPaths()

        # ROFF product information
        self.product_info = InSARProductsInfo.ROFF()


    def add_root_attrs(self):
        """
        Add root attributes
        """
        super().add_root_attrs()

        self.attrs["title"] = np.bytes_("NISAR L1 ROFF Product")
        self.attrs["reference_document"] = \
            np.bytes_("D-105009 NISAR NASA SDS"
                       " Product Specification L1 Range Doppler Pixel Offsets")

    def add_coregistration_to_algo_group(self):
        """
        Add the coregistration parameters to the "processingInfromation/algorithms" group
        """
        proc_cfg = self.cfg["processing"]
        dense_offsets = proc_cfg["dense_offsets"]["enabled"]
        offset_product = proc_cfg["offsets_product"]["enabled"]
        coreg_method = \
            "Coarse geometry coregistration with DEM and orbit ephemeris"
        if dense_offsets:
            coreg_method = f"{coreg_method} with cross-correlation refinement"

        algo_coregistration_ds_params = [
            DatasetParams(
                "coregistrationMethod",
                coreg_method,
                "RSLC coregistration method",
            ),
            DatasetParams(
                "geometryCoregistration",
                "Range doppler to geogrid then geogrid to range doppler"
                ,
                "Geometry coregistration algorithm",
            ),
            DatasetParams(
                "resampling",
                "sinc",
                "Secondary RSLC resampling algorithm",
            ),
        ]

        coregistration_group = self.require_group(
            f"{self.group_paths.AlgorithmsPath}/coregistration")
        for ds_param in algo_coregistration_ds_params:
            add_dataset_and_attrs(coregistration_group, ds_param)

    def add_cross_correlation_to_algo_group(self):
        """
        Add the cross correlation parameters to the "processingInfromation/algorithms" group
        """
        proc_cfg = self.cfg["processing"]
        is_roff = proc_cfg["offsets_product"]["enabled"]
        cross_correlation_domain = \
            get_off_params(proc_cfg, "cross_correlation_domain", is_roff)

        for layer in proc_cfg["offsets_product"].keys():
            if layer.startswith("layer"):
                cross_corr = DatasetParams(
                    "crossCorrelationAlgorithm",
                    cross_correlation_domain,
                    f"Cross-correlation algorithm for layer {layer[-1]}",
                )
                cross_corr_group = \
                    self.require_group(f"{self.group_paths.AlgorithmsPath}/crossCorrelation/{layer}")
                add_dataset_and_attrs(cross_corr_group, cross_corr)

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithms group to the processingInformation group
        """
        super().add_algorithms_to_procinfo_group()
        self.add_cross_correlation_to_algo_group()


    def add_parameters_to_procinfo_group(self):
        """
        Add parameters group to processingInformation/parameters group
        """
        # Using the InSARBase parameters group only
        InSARBaseWriter.add_parameters_to_procinfo_group(self)
        self.add_pixeloffsets_to_procinfo_params_group()

    def add_pixeloffsets_to_procinfo_params_group(self):
        """
        Add the pixelOffsets group to
        the processingInformation/parameters group
        """
        proc_cfg = self.cfg["processing"]
        # pull the offset parameters
        is_roff,  margin, rg_start, az_start,\
        rg_skip, az_skip, rg_search, az_search,\
        rg_chip, az_chip, ovs_factor = get_pixel_offsets_params(self.cfg)

        for freq, *_ in get_cfg_freq_pols(self.cfg):
            swath_frequency_path = \
                f"{self.ref_rslc.SwathPath}/frequency{freq}/"
            swath_frequency_group = self.ref_h5py_file_obj[
                swath_frequency_path]

            pixeloffsets_ds_params = [
                DatasetParams(
                    "alongTrackSkipWindowSize",
                    np.uint32(az_skip),
                    "Along-track cross-correlation skip window size in"
                    " pixels"
                    ,
                    {
                        "units": Units.unitless,
                    },
                ),
                DatasetParams(
                    "alongTrackStartPixel",
                    np.uint32(az_start),
                    "Reference RSLC start pixel in along-track",
                    {
                        "units": Units.unitless,
                    },
                ),
                DatasetParams(
                    "slantRangeSkipWindowSize",
                    np.uint32(rg_skip),
                    "Slant range cross-correlation skip window size in"
                    " pixels"
                    ,
                    {
                        "units": Units.unitless,
                    },
                ),
                DatasetParams(
                    "slantRangeStartPixel",
                    np.uint32(rg_start),
                    "Reference RSLC start pixel in slant range",
                    {
                        "units": Units.unitless,
                    },
                ),
                DatasetParams(
                    "correlationSurfaceOversampling",
                    np.uint32(ovs_factor),
                    "Oversampling factor of the cross-correlation surface"
                    ,
                    {
                        "units": Units.unitless,
                    },
                ),
                DatasetParams(
                    "margin",
                    np.uint32(margin),
                    "Margin in pixels around reference RSLC edges"
                    " excluded during cross-correlation"
                    " computation"
                    ,
                    {
                        "units": Units.unitless,
                    },
                ),
            ]

            pixeloffsets_group_name = \
                f"{self.group_paths.ParametersPath}/pixelOffsets/frequency{freq}"
            pixeloffsets_group = self.require_group(pixeloffsets_group_name)
            for ds_param in pixeloffsets_ds_params:
                add_dataset_and_attrs(pixeloffsets_group, ds_param)

            # TODO: the rangeBandwidth and azimuthBandwidth are placeholders heres,
            # and copied from the bandpassed RSLC data.
            # Should we update those fields?
            swath_frequency_group.copy(
                "processedRangeBandwidth",
                pixeloffsets_group,
                "rangeBandwidth",
            )
            swath_frequency_group.copy(
                "processedAzimuthBandwidth",
                pixeloffsets_group,
                "azimuthBandwidth",
            )

            for layer in proc_cfg["offsets_product"]:
                if layer.startswith("layer"):
                    rg_chip, az_chip, rg_search, az_search = \
                    [get_off_params(proc_cfg, off_param, is_roff,
                                    pattern=layer, get_min=True)
                     for off_param in ['window_range', 'window_azimuth',
                                       'half_search_range',
                                       'half_search_azimuth']]
                    ds_params = [
                        DatasetParams(
                            "alongTrackWindowSize",
                            np.uint32(az_chip),
                            "Along-track cross-correlation window size in"
                            " pixels"
                            ,
                            {
                                "units": Units.unitless,
                            },
                        ),
                        DatasetParams(
                            "slantRangeWindowSize",
                            np.uint32(rg_chip),
                            "Slant range cross-correlation window size in"
                            " pixels"
                            ,
                            {
                                "units": Units.unitless,
                            },
                        ),
                        DatasetParams(
                            "alongTrackSearchWindowSize",
                            np.uint32(2 * az_search),
                            "Along-track cross-correlation search window"
                            " size in pixels"
                            ,
                            {
                                "units": Units.unitless,
                            },
                        ),
                        DatasetParams(
                            "slantRangeSearchWindowSize",
                            np.uint32(2 * rg_search),
                            "Slant range cross-correlation search window"
                            " size in pixels"
                            ,
                            {
                                "units": Units.unitless,
                            },
                        ),
                    ]

                    layer_group_name = f"{pixeloffsets_group_name}/{layer}"
                    layer_group = self.require_group(layer_group_name)
                    for ds_param in ds_params:
                        add_dataset_and_attrs(layer_group, ds_param)

    def _add_datasets_to_pixel_offset_group(self):
        """
        Add datasets to pixelOffsets group under the swath group
        """
        # Add the ROFF specified datasets to the pixelOffset products
        proc_cfg = self.cfg["processing"]

        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            swaths_freq_group_name = (
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            )
            swaths_freq_group = self.require_group(swaths_freq_group_name)

            # shape of offset product
            off_shape = get_pixel_offsets_dataset_shape(self.cfg, freq)

            # pixel offsets dataset parameters including:
            # dataset name, description, and unit
            pixel_offsets_ds_params = [
                (
                    "alongTrackOffset",
                    "Raw (unculled, unfiltered) along-track pixel offsets",
                    Units.meter,
                ),
                (
                    "alongTrackOffsetVariance",
                    "Along-track pixel offsets variance",
                    Units.meter2,
                ),
                (
                    "slantRangeOffsetVariance",
                    "Slant range pixel offsets variance",
                    Units.meter2,
                ),
                (
                    "correlationSurfacePeak",
                    "Normalized correlation surface peak",
                    Units.unitless,
                ),
                (
                    "crossOffsetVariance",
                    "Off-diagonal term of the pixel offsets covariance matrix",
                    Units.meter2,
                ),
                (
                    "slantRangeOffset",
                    "Raw (unculled, unfiltered) slant range pixel offsets",
                    Units.meter,
                ),
                (
                    "snr",
                    "Pixel offsets signal-to-noise ratio",
                    Units.unitless,
                ),
            ]

            # add the list of layers
            self.add_list_of_layers(swaths_freq_group)

            # add the polarization dataset to pixelOffsets
            for pol in pol_list:
                offset_pol_group_name = \
                    f"{swaths_freq_group_name}/pixelOffsets/{pol}"
                pixeloffsets_pol_group = \
                    self.require_group(offset_pol_group_name)
                for layer in proc_cfg["offsets_product"]:
                    if layer.startswith("layer"):
                        layer_group_name = f"{offset_pol_group_name}/{layer}"
                        layer_group = self.require_group(layer_group_name)

                        # Create the pixel offsets dataset
                        for pixel_offsets_ds_param in pixel_offsets_ds_params:
                            ds_name, ds_description, ds_unit = pixel_offsets_ds_param
                            self._create_2d_dataset(
                                layer_group,
                                ds_name,
                                off_shape,
                                np.float32,
                                ds_description,
                                units=ds_unit,
                            )


    def add_list_of_layers(self, freq_group):
        """
        Get the requested layer groups from the runconfig,
        and add the `listOfLayers` Dataset to `freq_group`.

        Parameters
        ----------
        freq_group : h5py.Group
            The Group to add the `listOfLayers` Dataset to.
        """

        proc_cfg = self.cfg["processing"]

        # Extract offset layer group names, e.g. "layer1", "layer3".
        # If the layer group name appears in the runconfig,
        # that layer group will be generated, so add it to the list.
        layers = [
            layer
            for layer in proc_cfg["offsets_product"]
            if layer.startswith("layer")]

        list_of_layers = np.bytes_(layers)
        freq_group.require_dataset('listOfLayers',
                                    shape=list_of_layers.shape,
                                    dtype=list_of_layers.dtype,
                                    data=list_of_layers)

        freq_group['listOfLayers'].attrs['units'] = Units.unitless
        freq_group['listOfLayers'].attrs['description'] =\
            np.bytes_('List of pixel offsets layers')