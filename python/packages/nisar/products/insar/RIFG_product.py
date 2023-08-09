import os

import h5py
import numpy as np
from nisar.workflows.h5_prep import get_off_params
from nisar.workflows.helpers import get_cfg_freq_pols

from .common import ISCE3_VERSION, InSARProductsInfo
from .dataset_params import DatasetParams, add_dataset_and_attrs
from .product_paths import CommonPaths, RIFGGroupsPaths
from .InSARL1Products import L1InSARWriter

class RIFG(L1InSARWriter):
    """
    Writer class for RIFG product inherent from L1InSARWriter
    """

    def __init__(
        self,
        **kwds,
    ):
        """
        Constructor for RIFG class
        """
        super().__init__(**kwds)
        
    def save_to_hdf5(self):
        """
        write to the HDF5
        """
        
        self.add_root_attrs()
        self.add_identification_group()
        self.add_common_metadata_to_hdf5()
        self.add_geolocation_grid_cubes()
        self.add_processing_information_to_hdf5()
        self.add_swaths_to_hdf5()

    def _get_metadata_path(self):
        return RIFGGroupsPaths.MetadataPath

    def _get_geolocation_grid_cubes_path(self):
        return RIFGGroupsPaths.GeolocationGridPath

    def add_root_attrs(self):
        """
        add root attributes
        """
        
        super().add_root_attrs()
        
        self.attrs["title"] = "NISAR L1 RIFG Product"
        self.attrs["reference_document"] = "TBD"
        
        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(self["/"].id, np.string_("complex64"))
            
    def add_identification_group(self):
        
        super().add_identification_group()
        
        dst_id_group = self.require_group(CommonPaths.IdentificationPath)
        ds_params = [
            DatasetParams(
                "productVersion",
                InSARProductsInfo.RIFG().ProductVersion,
                (
                    "Product version which represents the structure of the"
                    " product and the science content governed by the"
                    " algorithm, input data, and processing parameters"
                ),
            ),
            DatasetParams("productType", InSARProductsInfo.RIFG().ProductType, "Product type"),
            DatasetParams(
                "productSpecificationVersion",
                InSARProductsInfo.RIFG().ProductSpecificationVersion,
                (
                    "Product specification version which represents the schema"
                    " of this product"
                ),
            ),
        ]

        for ds_param in ds_params:
            add_dataset_and_attrs(dst_id_group, ds_param)

    def _add_coregistration_to_algo_group(self, algo_group: h5py.Group):
        """
        add the coregistration parameters to the "processingInfromation/algorithms" group
        """

        pcfg = self.cfg["processing"]
        dense_offsets = pcfg["dense_offsets"]["enabled"]
        offset_product = pcfg["offsets_product"]["enabled"]

        if dense_offsets:
            name = "dense_offsets"
        elif offset_product:
            name = "offsets_product"

        coreg_method = (
            "Coarse geometry coregistration with DEM and orbit ephemeris"
        )

        cross_correlation_domain = "None"
        outlier_filling_method = "None"
        filter_kernel_algorithm = "None"
        culling_metric = "None"

        if dense_offsets or offset_product:
            coreg_method = f"{coreg_method} with cross-correlation refinement"
            cross_correlation_domain = pcfg[name]["cross_correlation_domain"]
            if pcfg["rubbersheet"]["enabled"]:
                outlier_filling_method = pcfg["rubbersheet"][
                    "outlier_filling_method"
                ]
                filter_kernel_algorithm = pcfg["rubbersheet"]["offsets_filter"]
                culling_metric = pcfg["rubbersheet"]["culling_metric"]

                # cross correlation filling
                if outlier_filling_method == "fill_smoothed":
                    description = (
                        "iterative filling algorithm using the mean value"
                        " computed in a neighboorhood centered on the pixel to"
                        " fill"
                    )
                else:
                    description = "Nearest neighboor interpolation"

        # Create list of DatasetParams to write to "algorithms/coregistration"
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
                "crossCorrelation",
                np.string_(cross_correlation_domain),
                np.string_(
                    "Cross-correlation algorithm for sub-pixel offsets"
                    f" computation in {cross_correlation_domain} domain"
                ),
                {
                    "algorithm_type": np.string_("RSLC coregistration"),
                },
            ),
            DatasetParams(
                "crossCorrelationFilling",
                np.string_(outlier_filling_method),
                np.string_(description),
                {
                    "algorithm_type": np.string_("RSLC coregistration"),
                },
            ),
            DatasetParams(
                "crossCorrelationFilterKernel",
                np.string_(filter_kernel_algorithm),
                np.string_(
                    "Filtering algorithm for cross-correlation offsets"
                ),
                {
                    "algorithm_type": np.string_("RSLC coregistration"),
                },
            ),
            DatasetParams(
                "crossCorrelationOutliers",
                np.string_(culling_metric),
                np.string_("Outliers identification algorithm"),
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

        # fill out above with remaining coregistration datasets
        coregistration_group = algo_group.require_group("coregistration")
        for ds_param in algo_coregistration_ds_params:
            add_dataset_and_attrs(coregistration_group, ds_param)

    def _add_interferogramformation_to_algo_group(self, algo_group):
        """
        add the InterferogramFormation to "processingInformation/algorithms" group
        """
        flatten_method = "None"
        pcfg = self.cfg["processing"]

        if pcfg["crossmul"]["flatten"]:
            flatten_method = "With geometry offsets"

        multilooking_method = "Spatial moving average with decimation"
        wrapped_interferogram_filtering_mdethod = pcfg["filter_interferogram"][
            "filter_type"
        ]

        algo_intefergramformation_ds_params = [
            DatasetParams(
                "flatteningMethod",
                np.string_(flatten_method),
                np.string_(
                    "Algorithm used to flatten the wrapped interferogram"
                ),
                {
                    #TODO: The description also needs to be changed in the product specs
                    "algorithm_type": np.string_("Interferogram formation"),
                },
            ),
            DatasetParams(
                "multilooking",
                np.string_(multilooking_method),
                np.string_("Multilooking algorithm"),
                {
                    "algorithm_type": np.string_("Interferogram formation"),
                },
            ),
            DatasetParams(
                "wrappedInterferogramFiltering",
                np.string_(wrapped_interferogram_filtering_mdethod),
                np.string_(
                    "Algorithm to filter wrapped interferogram prior to phase"
                    " unwrapping"
                ),
                {
                    "algorithm_type": np.string_("Interferogram formation"),
                },
            ),
        ]

        igram_formation_group = algo_group.require_group("interferogramFormation")
        
        for ds_param in algo_intefergramformation_ds_params:
            add_dataset_and_attrs(igram_formation_group, ds_param)

    def _add_algorithms_to_hdf5(self):
        """
        add the algorithms group to the processingInformation group
        """

        algo_group = self.require_group(RIFGGroupsPaths.AlgorithmsPath)
        self._add_coregistration_to_algo_group(algo_group)
        self._add_interferogramformation_to_algo_group(algo_group)

        software_version =  DatasetParams(
            "softwareVersion",
            np.string_(ISCE3_VERSION),
            np.string_("Software version used for processing"),
            )
        
        add_dataset_and_attrs(algo_group, software_version)

    def _add_inputs_to_hdf5(
        self,
        runconfig_path: str,
    ):
        """
        Add the inputs group to the "processingInformation" group
        """
        orbit_file = []
        for idx in ["reference", "secondary"]:
            _orbit_file = self.cfg["dynamic_ancillary_file_group"][
                "orbit"
            ].get(f"{idx}_orbit_file")
            if _orbit_file is None:
                _orbit_file = f"used RSLC internal {idx} orbit file"
            orbit_file.append(_orbit_file)

        # DEM source
        dem_source = self.cfg["dynamic_ancillary_file_group"].get(
            "dem_file_description"
        )

        # if dem source is None, then replace it with None
        if dem_source is None:
            dem_source = "None"

        inputs_ds_params = [
            DatasetParams(
                "configFiles",
                np.string_(os.path.basename(runconfig_path)),
                np.string_("List of input config files used"),
            ),
            DatasetParams(
                "demSource",
                np.string_(dem_source),
                np.string_(
                    "Description of the input digital elevation model (DEM)"
                ),
            ),
            DatasetParams(
                "l1ReferenceSlcGranules",
                np.string_([os.path.basename(self.ref_h5_slc_file)]),
                np.string_("List of input reference L1 RSLC products used"),
            ),
            DatasetParams(
                "l1SecondarySlcGranules",
                np.string_([os.path.basename(self.sec_h5_slc_file)]),
                np.string_("List of input secondary L1 RSLC products used"),
            ),
            DatasetParams(
                "orbitFiles",
                np.string_(orbit_file),
                np.string_("List of input orbit files used"),
            ),
        ]
        
        inputs_group = self.require_group(f"{RIFGGroupsPaths.ProcessingInformationPath}/inputs")
        for ds_param in inputs_ds_params:
            add_dataset_and_attrs(inputs_group, ds_param)

    def _add_parameters_to_hdf5(self):
        """
        Add the parameters group to the "processingInformation" group
        """
        params_group = self.require_group(RIFGGroupsPaths.ParametersPath)

        self._add_common_to_hdf5()
        self._add_interferogram_to_hdf5()
        self._add_pixeloffsets_to_hdf5()
        self._add_RSLC_to_hdf5("reference")
        self._add_RSLC_to_hdf5("secondary")
        
        runconfig_contents = DatasetParams(
            "runConfigurationContents",
            np.string_(str(self.cfg)),
            np.string_(
                "Contents of the run configuration file with parameters"
                " used for processing"
            ),
        )
        add_dataset_and_attrs(params_group, runconfig_contents)

    def _add_common_to_hdf5(self):
        """
        Add the common group to the "processingInformation/parameters" group
        """

        for freq, _, _ in get_cfg_freq_pols(self.cfg):
            doppler_centroid_path = f"{self.ref_rslc.ProcessingInformationPath}/parameters/frequency{freq}"
            doppler_bandwidth_path = (
                f"{self.ref_rslc.SwathPath}/frequency{freq}"
            )
            doppler_centroid_group = self.ref_h5py_file_obj[
                doppler_centroid_path
            ]
            doppler_bandwidth_group = self.ref_h5py_file_obj[
                doppler_bandwidth_path
            ]
            common_group_name = (
                f"{RIFGGroupsPaths.ParametersPath}/common/frequency{freq}"
            )
            common_group = self.require_group(common_group_name)

            self._copy_dataset_by_name(
                doppler_centroid_group, "dopplerCentroid", common_group
            )
            self._copy_dataset_by_name(
                doppler_bandwidth_group,
                "processedAzimuthBandwidth",
                common_group,
                "dopplerBandwidth",
            )

    def _add_interferogram_to_hdf5(self):
        """
        Add the interferogram to "processingInformation/parameters"
        """

        pcfg_crossmul = self.cfg["processing"]["crossmul"]
        range_filter = pcfg_crossmul["common_band_range_filter"]
        azimuth_filter = pcfg_crossmul["common_band_azimuth_filter"]

        flatten = pcfg_crossmul["flatten"]
        range_looks = pcfg_crossmul["range_looks"]
        azimuth_looks = pcfg_crossmul["azimuth_looks"]

        interferogram_ds_params = [
            DatasetParams(
                "commonBandRangeFilterApplied",
                np.bool_(range_filter),
                np.string_(
                    "Flag to indicate if common band range filter has been"
                    " applied"
                ),
            ),
            DatasetParams(
                "commonBandAzimuthFilterApplied",
                np.bool_(azimuth_filter),
                np.string_(
                    "Flag to indicate if common band azimuth filter has been"
                    " applied"
                ),
            ),
            DatasetParams(
                "ellipsoidalFlatteningApplied",
                np.bool_(flatten),
                np.string_(
                    "Flag to indicate if interferometric phase has been"
                    " flattened with respect to a zero height ellipsoid"
                ),
            ),
            DatasetParams(
                "topographicFlatteningApplied",
                np.bool_(flatten),
                np.string_(
                    "Flag to indicate if interferometric phase has been"
                    " flattened with respect to a zero height ellipsoid"
                ),
            ),
            DatasetParams(
                "numberOfRangeLooks",
                np.uint32(range_looks),
                np.string_(
                    "Number of looks applied in the slant range direction to"
                    " form the wrapped interferogram"
                ),
                {
                    "units": np.string_("unitless"),
                },
            ),
            DatasetParams(
                "numberOfAzimuthLooks",
                np.uint32(azimuth_looks),
                np.string_(
                    "Number of looks applied in the along-track direction to"
                    " form the wrapped interferogram"
                ),
                {
                    "units": np.string_("unitless"),
                },
            ),
        ]

        for freq, _, _ in get_cfg_freq_pols(self.cfg):
            bandwidth_group_path = f"{self.ref_rslc.SwathPath}/frequency{freq}"
            bandwidth_group = self.ref_h5py_file_obj[bandwidth_group_path]

            igram_group_name = f"{RIFGGroupsPaths.ParametersPath}/interferogram/frequency{freq}"
            igram_group = self.require_group(igram_group_name)

            self._copy_dataset_by_name(
                bandwidth_group,
                "processedAzimuthBandwidth",
                igram_group,
                "azimuthBandwidth",
            )
            self._copy_dataset_by_name(
                bandwidth_group,
                "processedRangeBandwidth",
                igram_group,
                "rangeBandwidth",
            )

            for ds_param in interferogram_ds_params:
                add_dataset_and_attrs(igram_group, ds_param)

    def _add_pixeloffsets_to_hdf5(self):
        """
        Add the pixelOffsets group to "processingInformation/parameters" group
        """

        pcfg = self.cfg["processing"]
        is_roff = pcfg["offsets_product"]["enabled"]
        merge_gross_offset = get_off_params(
            pcfg, "merge_gross_offset", is_roff
        )
        skip_range = get_off_params(pcfg, "skip_range", is_roff)
        skip_azimuth = get_off_params(pcfg, "skip_azimuth", is_roff)

        half_search_range = get_off_params(
            pcfg, "half_search_range", is_roff, pattern="layer", get_min=True
        )
        half_search_azimuth = get_off_params(
            pcfg, "half_search_azimuth", is_roff, pattern="layer", get_min=True
        )

        window_azimuth = get_off_params(
            pcfg, "window_azimuth", is_roff, pattern="layer", get_min=True
        )
        window_range = get_off_params(
            pcfg, "window_range", is_roff, pattern="layer", get_min=True
        )

        oversampling_factor = get_off_params(
            pcfg, "correlation_surface_oversampling_factor", is_roff
        )

        pixeloffsets_ds_params = [
            DatasetParams(
                "alongTrackWindowSize",
                np.uint32(window_azimuth),
                np.string_(
                    "Along track cross-correlation window size in pixels"
                ),
                {
                    "units": np.string_("unitless"),
                },
            ),
            DatasetParams(
                "slantRangeWindowSize",
                np.uint32(window_range),
                np.string_(
                    "Slant range cross-correlation window size in pixels"
                ),
                {
                    "units": np.string_("unitless"),
                },
            ),
            DatasetParams(
                "alongTrackSearchWindowSize",
                np.uint32(2 * half_search_azimuth),
                np.string_(
                    "Along track cross-correlation search window size in"
                    " pixels"
                ),
                {
                    "units": np.string_("unitless"),
                },
            ),
            DatasetParams(
                "slantRangeSearchWindowSize",
                np.uint32(2 * half_search_range),
                np.string_(
                    "Slant range cross-correlation search window size in"
                    " pixels"
                ),
                {
                    "units": np.string_("unitless"),
                },
            ),
            DatasetParams(
                "alongTrackSkipWindowSize",
                np.uint32(skip_azimuth),
                np.string_(
                    "Along track cross-correlation skip window size in pixels"
                ),
                {
                    "units": np.string_("unitless"),
                },
            ),
            DatasetParams(
                "slantRangeSkipWindowSize",
                np.uint32(skip_range),
                np.string_(
                    "Slant range cross-correlation skip window size in pixels"
                ),
                {
                    "units": np.string_("unitless"),
                },
            ),
            DatasetParams(
                "crossCorrelationSurfaceOversampling",
                np.uint32(oversampling_factor),
                np.string_(
                    "Oversampling factor of the cross-correlation surface"
                ),
                {
                    "units": np.string_("unitless"),
                },
            ),
            DatasetParams(
                "isOffsetsBlendingApplied",
                np.bool_(merge_gross_offset),
                np.string_(
                    "Flag to indicate if pixel offsets are the results of"
                    " blending multi-resolution layers of pixel offsets"
                ),
            ),
        ]
        for freq, _, _ in get_cfg_freq_pols(self.cfg):
            pixeloffsets_group_name = f"{RIFGGroupsPaths.ParametersPath}/pixelOffsets/frequency{freq}"
            pixeloffsets_group = self.require_group(pixeloffsets_group_name)

            for ds_param in pixeloffsets_ds_params:
                add_dataset_and_attrs(pixeloffsets_group, ds_param)

    def _add_RSLC_to_hdf5(self, name: str):
        """
        Add the RSLC to "processingInformation/parameters"
        """

        if name.lower() == "reference":
            h5py_file_obj = self.ref_h5py_file_obj
            rslc = self.ref_rslc
        else:
            h5py_file_obj = self.sec_h5py_file_obj
            rslc = self.sec_rslc

        try:
            rfi_mitigation = h5py_file_obj[
                f"{rslc.ProcessingInformationPath}/algorithms/rfiMitigation"
            ][()]
        except KeyError:
            # no RFI mitigation is found
            rfi_mitigation = None

        rfi_mitigation_flag = False
        if (rfi_mitigation is not None) and (rfi_mitigation != ""):
            rfi_mitigation_flag = True

        ds_params = [
            DatasetParams(
                "rfiCorrectionApplied",
                np.bool_(rfi_mitigation_flag),
                np.string_(
                    "Flag to indicate if RFI correction has been applied"
                    " to reference RSLC"
                ),
            ),
            self._get_mixed_mode(),
        ]

        group = self.require_group(f"{RIFGGroupsPaths.ParametersPath}/{name}")
        parameters_group = h5py_file_obj[
            f"{rslc.ProcessingInformationPath}/parameters"
        ]
        self._copy_dataset_by_name(
            parameters_group, "referenceTerrainHeight", group
        )
        for ds_param in ds_params:
            add_dataset_and_attrs(group, ds_param)

        for freq, _, _ in get_cfg_freq_pols(self.cfg):
            rslc_group_frequecy_name = (
                f"{RIFGGroupsPaths.ParametersPath}/{name}/frequency{freq}"
            )
            rslc_frequency_group = self.require_group(rslc_group_frequecy_name)

            swath_frequency_path = f"{rslc.SwathPath}/frequency{freq}/"
            swath_frequency_group = h5py_file_obj[swath_frequency_path]

            self._copy_dataset_by_name(
                swath_frequency_group,
                "slantRangeSpacing",
                rslc_frequency_group,
            )
            self._copy_dataset_by_name(
                swath_frequency_group,
                "processedRangeBandwidth",
                rslc_frequency_group,
                "rangeBandwidth",
            )
            self._copy_dataset_by_name(
                swath_frequency_group,
                "processedAzimuthBandwidth",
                rslc_frequency_group,
                "azimuthBandwidth",
            )

            swath_group = h5py_file_obj[rslc.SwathPath]
            self._copy_dataset_by_name(
                swath_group, "zeroDopplerTimeSpacing", rslc_frequency_group
            )

            doppler_centroid_group = h5py_file_obj[
                f"{rslc.ProcessingInformationPath}/parameters/frequency{freq}"
            ]
            self._copy_dataset_by_name(
                doppler_centroid_group, "dopplerCentroid", rslc_frequency_group
            )

    def add_processing_information_to_hdf5(self):
        """
        Add processing information group to metadata
        """
        self._add_algorithms_to_hdf5()
        self._add_inputs_to_hdf5(self.runconfig_path)
        self._add_parameters_to_hdf5()

    def add_swaths_to_hdf5(self):
        """
        Add Swaths to the HDF5
        """
        pcfg = self.cfg["processing"]
        rg_looks = pcfg["crossmul"]["range_looks"]
        az_looks = pcfg["crossmul"]["azimuth_looks"]

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
                f"{RIFGGroupsPaths.SwathsPath}/frequency{freq}"
            )
            swaths_freq_group = self.require_group(swaths_freq_group_name)

            # Create the interferogram and pixeloffsets group
            igram_group_name = f"{swaths_freq_group_name}/interferogram"
            offset_group_name = f"{swaths_freq_group_name}/pixelOffsets"

            igram_group = self.require_group(igram_group_name)
            offset_group = self.require_group(offset_group_name)

            # center frequency and sub swaths groups of the RSLC
            rslc_swaths_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}"
            ]
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

            # shape of the interferogram product
            igram_shape = (slc_lines // az_looks, slc_cols // rg_looks)

            self._copy_dataset_by_name(
                rslc_freq_group,
                "processedCenterFrequency",
                swaths_freq_group,
                "centerFrequency",
            )
            self._copy_dataset_by_name(
                rslc_freq_group, "numberOfSubSwaths", swaths_freq_group
            )

            scence_center_params = [
                DatasetParams(
                    "sceneCenterAlongTrackSpacing",
                    rslc_freq_group["sceneCenterAlongTrackSpacing"][()]
                    * az_looks,
                    np.string_(
                        "Nominal along track spacing in meters between"
                        " consecutive lines near mid swath of the RIFG image"
                    ),
                    {"units": np.string_("meters")},
                ),
                DatasetParams(
                    "sceneCenterGroundRangeSpacing",
                    rslc_freq_group["sceneCenterGroundRangeSpacing"][()]
                    * rg_looks,
                    np.string_(
                        "Nominal ground range spacing in meters between"
                        " consecutive pixels near mid swath of the RIFG image"
                    ),
                    {"units": np.string_("meters")},
                ),
            ]
            for ds_param in scence_center_params:
                add_dataset_and_attrs(swaths_freq_group, ds_param)

            # valid samples subswath
            num_of_subswaths = rslc_freq_group["numberOfSubSwaths"][()]
            for sub in range(num_of_subswaths):
                subswath = sub + 1
                valid_samples_subswath_name = f"validSamplesSubSwath{subswath}"
                if valid_samples_subswath_name in rslc_freq_group.keys():
                    number_of_range_looks = (
                        rslc_freq_group[valid_samples_subswath_name][()]
                        // rg_looks
                    )
                    swaths_freq_group.require_dataset(
                        name=valid_samples_subswath_name,
                        data=number_of_range_looks,
                        shape=number_of_range_looks.shape,
                        dtype=number_of_range_looks.dtype,
                    )
                    swaths_freq_group[
                        valid_samples_subswath_name
                    ].attrs.update(
                        rslc_freq_group[valid_samples_subswath_name].attrs
                    )
                else:
                    number_of_range_looks = (
                        rslc_freq_group["validSamples"][()] // rg_looks
                    )
                    swaths_freq_group.require_dataset(
                        name="validSamples",
                        data=number_of_range_looks,
                        shape=number_of_range_looks.shape,
                        dtype=number_of_range_looks.dtype,
                    )
                    swaths_freq_group["validSamples"].attrs.update(
                        rslc_freq_group["validSamples"].attrs
                    )

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

            #  add the slantRange, zeroDopplerTime, and their spacings to inteferogram group
            igram_slant_range = rslc_freq_group["slantRange"][()]
            igram_zero_doppler_time = rslc_swaths_group["zeroDopplerTime"][()]
            rg_idx = np.arange(
                (len(igram_slant_range) // rg_looks) * rg_looks
            )[::rg_looks] + int(rg_looks / 2)
            az_idx = np.arange(
                (len(igram_zero_doppler_time) // az_looks) * az_looks
            )[::az_looks] + int(az_looks / 2)

            igram_slant_range = igram_slant_range[rg_idx]
            igram_zero_doppler_time = igram_zero_doppler_time[az_idx]

            igram_group.require_dataset(
                name="slantRange",
                data=igram_slant_range,
                shape=igram_slant_range.shape,
                dtype=igram_slant_range.dtype,
            )
            igram_group["slantRange"].attrs.update(
                rslc_freq_group["slantRange"].attrs
            )
            igram_group["slantRange"].attrs["description"] = np.string_(
                "Slant range vector"
            )

            igram_zero_doppler_time = rslc_swaths_group["zeroDopplerTime"][()]
            igram_group.require_dataset(
                name="zeroDopplerTime",
                data=igram_zero_doppler_time,
                shape=igram_zero_doppler_time.shape,
                dtype=igram_zero_doppler_time.dtype,
            )
            igram_group["zeroDopplerTime"].attrs.update(
                rslc_swaths_group["zeroDopplerTime"].attrs
            )
            igram_group["zeroDopplerTime"].attrs["description"] = np.string_(
                "Zero Doppler azimuth time vector"
            )

            igram_zero_doppler_time_spacing = (
                rslc_swaths_group["zeroDopplerTimeSpacing"][()] * az_looks
            )
            igram_group.require_dataset(
                name="zeroDopplerTimeSpacing",
                data=igram_zero_doppler_time_spacing,
                shape=igram_zero_doppler_time_spacing.shape,
                dtype=igram_zero_doppler_time_spacing.dtype,
            )
            igram_group["zeroDopplerTimeSpacing"].attrs.update(
                rslc_swaths_group["zeroDopplerTimeSpacing"].attrs
            )

            igram_slant_range_spacing = (
                rslc_freq_group["slantRangeSpacing"][()] * rg_looks
            )
            igram_group.require_dataset(
                name="slantRangeSpacing",
                data=igram_slant_range_spacing,
                shape=igram_slant_range_spacing.shape,
                dtype=igram_slant_range_spacing.dtype,
            )
            igram_group["slantRangeSpacing"].attrs.update(
                rslc_freq_group["slantRangeSpacing"].attrs
            )

            # add the polarization
            for pol in pol_list:
                igram_pol_group_name = (
                    f"{swaths_freq_group_name}/interferogram/{pol}"
                )
                offset_pol_group_name = (
                    f"{swaths_freq_group_name}/pixelOffsets/{pol}"
                )

                igram_pol_group = self.require_group(igram_pol_group_name)
                offset_pol_group = self.require_group(offset_pol_group_name)

                # Create the inteferogram dataset
                self._create_2d_dataset(
                    igram_pol_group,
                    "coherenceMagnitude",
                    igram_shape,
                    np.float32,
                    np.string_(f"Coherence magnitude between {pol} layers"),
                    units=np.string_("unitless"),
                )
                self._create_2d_dataset(
                    igram_pol_group,
                    "wrappedInterferogram",
                    igram_shape,
                    np.complex64,
                    np.string_(f"Interferogram between {pol} layers"),
                    units=np.string_("DN"),
                )

                # Create the pixel offsets dataset
                self._create_2d_dataset(
                    offset_pol_group,
                    "alongTrackOffset",
                    off_shape,
                    np.float32,
                    np.string_(f"Along track offset"),
                    units=np.string_("meters"),
                )
                self._create_2d_dataset(
                    offset_pol_group,
                    "crossCorrelationPeak",
                    off_shape,
                    np.float32,
                    np.string_(f"Normalized cross-correlation surface peak"),
                    units=np.string_("unitless"),
                )
                self._create_2d_dataset(
                    offset_pol_group,
                    "slantRangeOffset",
                    off_shape,
                    np.float32,
                    np.string_(f"Slant range offset"),
                    units=np.string_("meters"),
                )
