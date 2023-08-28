import h5py
import numpy as np
from nisar.workflows.h5_prep import get_off_params
from nisar.workflows.helpers import get_cfg_freq_pols

from .common import InSARProductsInfo
from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_base_writer import InSARWriter
from .InSAR_L1_writer import L1InSARWriter
from .product_paths import ROFFGroupsPaths


class ROFFWriter(L1InSARWriter):
    """
    Writer class for ROFF product inherent from L1InSARWriter
    """

    def __init__(
        self,
        **kwds,
    ):
        """
        Constructor for ROFF class
        """
        super().__init__(**kwds)

        # group paths are ROFF group paths
        self.group_paths = ROFFGroupsPaths()

        # ROFF product information
        self.product_info = InSARProductsInfo.ROFF()

    def add_root_attrs(self):
        """
        add root attributes
        """

        super().add_root_attrs()

        self.attrs["title"] = "NISAR L1 ROFF Product"
        self.attrs["reference_document"] = "TBD"

    def add_coregistration_to_algo(self, algo_group: h5py.Group):
        """
        Add the coregistration parameters to the "processingInfromation/algorithms" group

        Parameters
        ------
        - algo_group (h5py.Group): the algorithm group object

        Return
        ------
        - coregistration_group (h5py.Group): the coregistration group object
        """

        pcfg = self.cfg["processing"]
        dense_offsets = pcfg["dense_offsets"]["enabled"]
        offset_product = pcfg["offsets_product"]["enabled"]

        coreg_method = (
            "Coarse geometry coregistration with DEM and orbit ephemeris"
        )
        if dense_offsets or offset_product:
            coreg_method = f"{coreg_method} with cross-correlation refinement"

        algo_coregistration_ds_params = [
            DatasetParams(
                "coregistrationMethod",
                np.string_(coreg_method),
                np.string_("RSLC coregistration method"),
                {
                    "algorithm_type": np.string_("RSLC coregistration"),
                },
            ),
            DatasetParams(
                "geometryCoregistration",
                np.string_(
                    "Range doppler to geogrid then geogrid to range doppler"
                ),
                np.string_("Geometry coregistration algorithm"),
                {
                    "algorithm_type": np.string_("RSLC coregistration"),
                },
            ),
            DatasetParams(
                "resampling",
                np.string_("sinc"),
                np.string_("Secondary RSLC resampling algorithm"),
                {
                    "algorithm_type": np.string_("RSLC coregistration"),
                },
            ),
        ]

        coregistration_group = algo_group.require_group("coregistration")
        for ds_param in algo_coregistration_ds_params:
            add_dataset_and_attrs(coregistration_group, ds_param)

        return coregistration_group

    def add_interferogramformation_to_algo(self, algo_group: h5py.Group):
        """
        Add the interferogram information to algorithm group.

        NOTE: Since there is no interferogram information in the ROFF algorithm group,
        and the add_algorithms_to_procinfo() function of the L1InSARWriter class has added this group
        to the algorithm group, to avoid adding this group to ROFF product,
        we will leave this function doing nothing here.
        """

        pass

    def add_cross_correlation_to_algo(self, algo_group: h5py.Group):
        """
        Add the cross correlation parameters to the "processingInfromation/algorithms" group

        Parameters
        ------
        - algo_group (h5py.Group): the algorithm group object
        """

        pcfg = self.cfg["processing"]
        is_roff = pcfg["offsets_product"]["enabled"]

        cross_correlation_domain = get_off_params(
            pcfg, "cross_correlation_domain", is_roff
        )

        for layer in pcfg["offsets_product"].keys():
            if layer.startswith("layer"):
                # layer specific cross correlation domain
                cross_correlation_domain = get_off_params(
                    pcfg, "cross_correlation_domain", is_roff, pattern=layer
                )
                cross_corr = DatasetParams(
                    "crossCorrelationAlgorithm",
                    np.string_(cross_correlation_domain),
                    np.string_(
                        f"Cross-correlation algorithm for layer {layer[-1]}"
                    ),
                    {
                        "algorithm_type": np.string_("RSLC coregistration"),
                    },
                )
                cross_corr_group = algo_group.require_group(
                    f"crossCorrelation/{layer}"
                )
                add_dataset_and_attrs(cross_corr_group, cross_corr)

    def add_algorithms_to_procinfo(self):
        """
        Add the algorithms to processingInformation group

        Return
        ------
        algo_group (h5py.Group): the algorithm group object
        """

        algo_group = super().add_algorithms_to_procinfo()
        self.add_cross_correlation_to_algo(algo_group)

        return algo_group

    def add_parameters_to_procinfo(self):
        """
        Add parameters group to processingInformation/parameters group
        """

        # Using the InSARBase parameters group only
        InSARWriter.add_parameters_to_procinfo(self)
        self.add_pixeloffsets_to_procinfo_params()

    def add_pixeloffsets_to_procinfo_params(self):
        """
        Add the pixelOffsets to the processingInformation/parameters group
        """

        # pull the offset parameters
        pcfg = self.cfg["processing"]
        is_roff = pcfg["offsets_product"]["enabled"]
        margin = get_off_params(pcfg, "margin", is_roff)
        rg_gross = get_off_params(pcfg, "gross_offset_range", is_roff)
        az_gross = get_off_params(pcfg, "gross_offset_azimuth", is_roff)
        rg_start = get_off_params(pcfg, "start_pixel_range", is_roff)
        az_start = get_off_params(pcfg, "start_pixel_azimuth", is_roff)
        rg_skip = get_off_params(pcfg, "skip_range", is_roff)
        az_skip = get_off_params(pcfg, "skip_azimuth", is_roff)
        rg_search = get_off_params(
            pcfg, "half_search_range", is_roff, pattern="layer", get_min=True
        )
        az_search = get_off_params(
            pcfg, "half_search_azimuth", is_roff, pattern="layer", get_min=True
        )
        ovs_factor = get_off_params(
            pcfg, "correlation_surface_oversampling_factor", is_roff
        )
        # Adjust margin
        margin = max(margin, np.abs(rg_gross), np.abs(az_gross))

        # Compute slant range/azimuth vectors of offset grids
        if rg_start is None:
            rg_start = margin + rg_search
        if az_start is None:
            az_start = margin + az_search

        for freq, _, _ in get_cfg_freq_pols(self.cfg):
            swath_frequency_path = (
                f"{self.ref_rslc.SwathPath}/frequency{freq}/"
            )
            swath_frequency_group = self.ref_h5py_file_obj[
                swath_frequency_path
            ]

            pixeloffsets_ds_params = [
                DatasetParams(
                    "alongTrackSkipWindowSize",
                    np.uint32(az_skip),
                    np.string_(
                        "Along track cross-correlation skip window size in"
                        " pixels"
                    ),
                    {
                        "units": np.string_("unitless"),
                    },
                ),
                DatasetParams(
                    "alongTrackStartPixel",
                    np.uint32(az_start),
                    np.string_("Reference RSLC start pixel in along track"),
                    {
                        "units": np.string_("unitless"),
                    },
                ),
                DatasetParams(
                    "slantRangeSkipWindowSize",
                    np.uint32(rg_skip),
                    np.string_(
                        "Slant range cross-correlation skip window size in"
                        " pixels"
                    ),
                    {
                        "units": np.string_("unitless"),
                    },
                ),
                DatasetParams(
                    "slantRangeStartPixel",
                    np.uint32(rg_start),
                    np.string_("Reference RSLC start pixel in slant range"),
                    {
                        "units": np.string_("unitless"),
                    },
                ),
                DatasetParams(
                    "crossCorrelationSurfaceOversampling",
                    np.uint32(ovs_factor),
                    np.string_(
                        "Oversampling factor of the cross-correlation surface"
                    ),
                    {
                        "units": np.string_("unitless"),
                    },
                ),
                DatasetParams(
                    "margin",
                    np.uint32(margin),
                    np.string_(
                        "Margin in pixels around reference RSLC edges"
                        " excluded during cross-correlation"
                        " computation"
                    ),
                    {
                        "units": np.string_("unitless"),
                    },
                ),
            ]

            pixeloffsets_group_name = f"{self.group_paths.ParametersPath}/pixelOffsets/frequency{freq}"
            pixeloffsets_group = self.require_group(pixeloffsets_group_name)
            for ds_param in pixeloffsets_ds_params:
                add_dataset_and_attrs(pixeloffsets_group, ds_param)

            self._copy_dataset_by_name(
                swath_frequency_group,
                "processedRangeBandwidth",
                pixeloffsets_group,
                "rangeBandwidth",
            )
            self._copy_dataset_by_name(
                swath_frequency_group,
                "processedAzimuthBandwidth",
                pixeloffsets_group,
                "azimuthBandwidth",
            )

            for layer in pcfg["offsets_product"]:
                if layer.startswith("layer"):
                    rg_chip = get_off_params(
                        pcfg,
                        "window_range",
                        is_roff,
                        pattern=layer,
                        get_min=True,
                    )
                    az_chip = get_off_params(
                        pcfg,
                        "window_azimuth",
                        is_roff,
                        pattern=layer,
                        get_min=True,
                    )
                    rg_search = get_off_params(
                        pcfg,
                        "half_search_range",
                        is_roff,
                        pattern=layer,
                        get_min=True,
                    )
                    az_search = get_off_params(
                        pcfg,
                        "half_search_azimuth",
                        is_roff,
                        pattern=layer,
                        get_min=True,
                    )

                    ds_params = [
                        DatasetParams(
                            "alongTrackWindowSize",
                            np.uint32(az_chip),
                            np.string_(
                                "Along track cross-correlation window size in"
                                " pixels"
                            ),
                            {
                                "units": np.string_("unitless"),
                            },
                        ),
                        DatasetParams(
                            "slantRangeWindowSize",
                            np.uint32(rg_chip),
                            np.string_(
                                "Slant range cross-correlation window size in"
                                " pixels"
                            ),
                            {
                                "units": np.string_("unitless"),
                            },
                        ),
                        DatasetParams(
                            "alongTrackSearchWindowSize",
                            np.uint32(2 * az_search),
                            np.string_(
                                "Along track cross-correlation search window"
                                " size in pixels"
                            ),
                            {
                                "units": np.string_("unitless"),
                            },
                        ),
                        DatasetParams(
                            "slantRangeSearchWindowSize",
                            np.uint32(2 * rg_search),
                            np.string_(
                                "Slant range cross-correlation search window"
                                " size in pixels"
                            ),
                            {
                                "units": np.string_("unitless"),
                            },
                        ),
                    ]

                    layer_group_name = f"{pixeloffsets_group_name}/{layer}"
                    layer_group = self.require_group(layer_group_name)
                    for ds_param in ds_params:
                        add_dataset_and_attrs(layer_group, ds_param)

    def add_swaths_to_hdf5(self):
        """
        Add Swaths to the HDF5
        """

        super().add_swaths_to_hdf5()

        pcfg = self.cfg["processing"]

        # pull the offset parameters
        is_roff = pcfg["offsets_product"]["enabled"]
        margin = get_off_params(pcfg, "margin", is_roff)
        rg_gross = get_off_params(pcfg, "gross_offset_range", is_roff)
        az_gross = get_off_params(pcfg, "gross_offset_azimuth", is_roff)
        rg_start = get_off_params(pcfg, "start_pixel_range", is_roff)
        az_start = get_off_params(pcfg, "start_pixel_azimuth", is_roff)
        rg_skip = get_off_params(pcfg, "skip_range", is_roff)
        az_skip = get_off_params(pcfg, "skip_azimuth", is_roff)
        rg_search = get_off_params(
            pcfg,
            "half_search_range",
            is_roff,
            pattern="layer",
            get_min=True,
        )
        az_search = get_off_params(
            pcfg,
            "half_search_azimuth",
            is_roff,
            pattern="layer",
            get_min=True,
        )
        rg_chip = get_off_params(
            pcfg, "window_range", is_roff, pattern="layer", get_min=True
        )
        az_chip = get_off_params(
            pcfg, "window_azimuth", is_roff, pattern="layer", get_min=True
        )
        # Adjust margin
        margin = max(margin, np.abs(rg_gross), np.abs(az_gross))

        # Compute slant range/azimuth vectors of offset grids
        if rg_start is None:
            rg_start = margin + rg_search
        if az_start is None:
            az_start = margin + az_search

        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            swaths_freq_group_name = (
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            )
            self.require_group(swaths_freq_group_name)

            # Createpixeloffsets group
            offset_group_name = f"{swaths_freq_group_name}/pixelOffsets"

            offset_group = self.require_group(offset_group_name)

            # center frequency and sub swaths groups of the RSLC
            rslc_swaths_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}"
            ]
            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"
            ]

            # get the RSLC lines and columns
            slc_dset = self.ref_h5py_file_obj[
                f'{f"{self.ref_rslc.SwathPath}/frequency{freq}"}/{pol_list[0]}'
            ]
            slc_lines, slc_cols = slc_dset.shape

            off_length = get_off_params(pcfg, "offset_length", is_roff)
            off_width = get_off_params(pcfg, "offset_width", is_roff)
            if off_length is None:
                margin_az = 2 * margin + 2 * az_search + az_chip
                off_length = (slc_lines - margin_az) // az_skip
            if off_width is None:
                margin_rg = 2 * margin + 2 * rg_search + rg_chip
                off_width = (slc_cols - margin_rg) // rg_skip

            # shape of offset product
            off_shape = (off_length, off_width)

            # add the slantRange, zeroDopplerTime, and their spacings to pixel offset group
            offset_slant_range = rslc_freq_group["slantRange"][()][
                rg_start::rg_skip
            ][:off_width]
            offset_group.require_dataset(
                name="slantRange",
                data=offset_slant_range,
                shape=offset_slant_range.shape,
                dtype=offset_slant_range.dtype,
            )
            offset_group["slantRange"].attrs.update(
                rslc_freq_group["slantRange"].attrs
            )
            offset_group["slantRange"].attrs["description"] = np.string_(
                "Slant range vector"
            )

            offset_zero_doppler_time = rslc_swaths_group["zeroDopplerTime"][
                ()
            ][az_start::az_skip][:off_length]
            offset_group.require_dataset(
                name="zeroDopplerTime",
                data=offset_zero_doppler_time,
                shape=offset_zero_doppler_time.shape,
                dtype=offset_zero_doppler_time.dtype,
            )
            offset_group["zeroDopplerTime"].attrs.update(
                rslc_swaths_group["zeroDopplerTime"].attrs
            )
            offset_group["zeroDopplerTime"].attrs["description"] = np.string_(
                "Zero Doppler azimuth time vector"
            )

            offset_zero_doppler_time_spacing = (
                rslc_swaths_group["zeroDopplerTimeSpacing"][()] * az_skip
            )
            offset_group.require_dataset(
                name="zeroDopplerTimeSpacing",
                data=offset_zero_doppler_time_spacing,
                shape=offset_zero_doppler_time_spacing.shape,
                dtype=offset_zero_doppler_time_spacing.dtype,
            )
            offset_group["zeroDopplerTimeSpacing"].attrs.update(
                rslc_swaths_group["zeroDopplerTimeSpacing"].attrs
            )

            offset_slant_range_spacing = (
                rslc_freq_group["slantRangeSpacing"][()] * rg_skip
            )
            offset_group.require_dataset(
                name="slantRangeSpacing",
                data=offset_slant_range_spacing,
                shape=offset_slant_range_spacing.shape,
                dtype=offset_slant_range_spacing.dtype,
            )
            offset_group["slantRangeSpacing"].attrs.update(
                rslc_freq_group["slantRangeSpacing"].attrs
            )

            # add the polarization
            for pol in pol_list:
                offset_pol_group_name = (
                    f"{swaths_freq_group_name}/pixelOffsets/{pol}"
                )
                self.require_group(offset_pol_group_name)

                for layer in pcfg["offsets_product"]:
                    if layer.startswith("layer"):
                        layer_group_name = f"{offset_pol_group_name}/{layer}"
                        layer_group = self.require_group(layer_group_name)

                        # Create the pixel offsets dataset
                        self._create_2d_dataset(
                            layer_group,
                            "alongTrackOffset",
                            off_shape,
                            np.float32,
                            np.string_(f"Along track offset"),
                            units=np.string_("meters"),
                        )
                        self._create_2d_dataset(
                            layer_group,
                            "alongTrackOffsetVariance",
                            off_shape,
                            np.float32,
                            np.string_(f"Along-track pixel offsets variance"),
                            np.string_("unitless"),
                        )
                        self._create_2d_dataset(
                            layer_group,
                            "slantRangeOffsetVariance",
                            off_shape,
                            np.float32,
                            np.string_(f"Slant range pixel offsets variance"),
                            np.string_("unitless"),
                        )
                        self._create_2d_dataset(
                            layer_group,
                            "crossCorrelationPeak",
                            off_shape,
                            np.float32,
                            np.string_(
                                f"Normalized cross-correlation surface peak"
                            ),
                            units=np.string_("unitless"),
                        )
                        self._create_2d_dataset(
                            layer_group,
                            "crossOffsetVariance",
                            off_shape,
                            np.float32,
                            np.string_(
                                f"Off-diagonal term of the pixel offsets"
                                f" covariance matrix"
                            ),
                            units=np.string_("unitless"),
                        )
                        self._create_2d_dataset(
                            layer_group,
                            "slantRangeOffset",
                            off_shape,
                            np.float32,
                            np.string_(f"Slant range offset"),
                            units=np.string_("meters"),
                        )
                        self._create_2d_dataset(
                            layer_group,
                            "snr",
                            off_shape,
                            np.float32,
                            np.string_(f"Pixel offsets signal-to-noise ratio"),
                            units=np.string_("unitless"),
                        )
