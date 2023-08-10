import os
from datetime import datetime
from typing import Any, Optional

import h5py
import numpy as np
from isce3.core import DateTime
from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows.h5_prep import get_off_params
from nisar.workflows.helpers import get_cfg_freq_pols

from .common import ISCE3_VERSION, InSARProductsInfo, get_validated_file_path
from .dataset_params import DatasetParams, add_dataset_and_attrs
from .product_paths import CommonPaths


class InSARWriter(h5py.File):
    """
    The base class of InSAR product inheriting from h5py.File to avoid passing
    h5py.File parameter

    Attributes:
    ------
    - cfg (dict): runconfig dictionrary
    - runconfig_path (str):  path of the runconfig file
    - ref_h5_slc_file (str): path of the reference RSLC
    - sec_h5_slc_file (str): path of the secondary RSLC
    - external_orbit_path (str): path of the external orbit file
    - epoch (Datetime): the reference datetime for the orbit
    - ref_rslc (object): SLC object of reference RSLC file
    - sec_rslc (object): SLC object of the secondary RSLC file
    - ref_h5py_file_obj (h5py.File): h5py object of the reference RSLC
    - sec_h5py_file_obj (h5py.File): h5py object of the secondary RSLC
    - kwds: parameters of the h5py.File
    """

    def __init__(
        self,
        runconfig_dict: dict,
        runconfig_path: str,
        _external_orbit_path: Optional[str] = None,
        epoch: Optional[DateTime] = None,
        **kwds,
    ):
        """
        Constructor of the InSAR Product Base class. Inheriting from h5py.File
        to avoid passing h5py.File parameter.

        Parameters
        ------
        - runconfig_dict (dict): runconfig dictionrary
        - runconfig_path (str): path of the reference RSLC
        - external_orbit_path (Optional[str]): path of the external orbit file
        - epoch (Optional[Datetime]): the reference datetime for the orbit
        """

        super().__init__(**kwds)

        self.cfg = runconfig_dict
        self.runconfig_path = runconfig_path

        # Reference and Secondary RSLC files
        self.ref_h5_slc_file = self.cfg["input_file_group"][
            "reference_rslc_file"
        ]
        self.sec_h5_slc_file = self.cfg["input_file_group"][
            "secondary_rslc_file"
        ]

        # Pull the frequency and polarizations
        self.freq_pols = self.cfg["processing"]["input_subset"][
            "list_of_frequencies"
        ]

        # group paths
        self.group_paths = CommonPaths()

        # product information
        self.product_info = InSARProductsInfo.Base()
        
        # Epoch time
        self.epoch = epoch

        # Check if reference and secondary exists as files
        [
            self.path_ref_slc_h5,
            self.path_sec_slc_h5,
            self.external_orbit_path,
        ] = [
            get_validated_file_path(path_str)
            for path_str in [
                self.ref_h5_slc_file,
                self.sec_h5_slc_file,
                _external_orbit_path,
            ]
        ]

        self.ref_rslc = SLC(hdf5file=self.ref_h5_slc_file)
        self.sec_rslc = SLC(hdf5file=self.sec_h5_slc_file)
        
        # Open the reference file
        try:
            self.ref_h5py_file_obj = h5py.File(
                self.ref_h5_slc_file, "r", libver="latest", swmr=True
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot Open the {self.ref_h5_slc_file} file")
        
        # Open the secondary file
        try:
            self.sec_h5py_file_obj = h5py.File(
                self.sec_h5_slc_file, "r", libver="latest", swmr=True
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot Open the {self.sec_h5_slc_file} file")

    def add_root_attrs(self):
        """
        Write attributes to root that are common to all InSAR products
        """
        self.attrs["Conventions"] = np.string_("CF-1.7")
        self.attrs["contact"] = np.string_("nisarops@jpl.nasa.gov")
        self.attrs["institution"] = np.string_("NASA JPL")
        self.attrs["mission_name"] = np.string_("NISAR")

    def save_to_hdf5(self):
        """
        Write to the HDF5
        """
        self.add_root_attrs()
        self.add_identification_to_hdf5()
        self.add_common_metadata_to_hdf5()
        self.add_procinfo_to_metadata()

    def add_procinfo_to_metadata(self):
        """
        Add processing information to metadata
        
        Return
        ------
        group (h5py.Group): the processing information group object
        """
        group = self.require_group(self.group_paths.ProcessingInformationPath)
        self.add_algorithms_to_procinfo()
        self.add_inputs_to_procinfo(self.runconfig_path)
        self.add_parameters_to_procinfo()
        
        return group
    
    def add_algorithms_to_procinfo(self):
        """
        Add the algorithm to the processing information
        
        Return
        ------
        algo_group (h5py.Group): the algorithm group object
        """

        algo_group = self.require_group(self.group_paths.AlgorithmsPath)

        software_version =  DatasetParams(
            "softwareVersion",
            np.string_(ISCE3_VERSION),
            np.string_("Software version used for processing"),
            )
        
        add_dataset_and_attrs(algo_group, software_version)
        
        return algo_group
        
    def add_common_to_procinfo_params(self):
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
                f"{self.group_paths.ParametersPath}/common/frequency{freq}"
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

    def add_RSLC_to_procinfo_params(self, name: str):
        """
        Add the RSLC to "processingInformation/parameters"
        
        Return
        ------
        group (h5py.Group): the RSLC group object
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

        group = self.require_group(f"{self.group_paths.ParametersPath}/{name}")
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
                f"{self.group_paths.ParametersPath}/{name}/frequency{freq}"
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
            
        return group
        
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

                if outlier_filling_method == "fill_smoothed":
                    description = (
                        "iterative filling algorithm using the mean value"
                        " computed in a neighboorhood centered on the pixel to"
                        " fill"
                    )
                else:
                    description = "Nearest neighboor interpolation"

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
        
        coregistration_group = algo_group.require_group("coregistration")
        for ds_param in algo_coregistration_ds_params:
            add_dataset_and_attrs(coregistration_group, ds_param)
            
        return coregistration_group

    def add_interferogramformation_to_algo(self, algo_group: h5py.Group):
        """
        Add the InterferogramFormation to "processingInformation/algorithms" group
        
        Parameters
        ------
        - algo_group (h5py.Group): the algorithm group object
        
        Return
        ------
        - igram_formation_group (h5py.Group): the interfergram formation group object
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
            
    def add_interferogram_to_procinfo_params(self):
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

            igram_group_name = f"{self.group_paths.ParametersPath}/interferogram/frequency{freq}"
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

    def add_pixeloffsets_to_procinfo_params(self):
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
            pixeloffsets_group_name = f"{self.group_paths.ParametersPath}/pixelOffsets/frequency{freq}"
            pixeloffsets_group = self.require_group(pixeloffsets_group_name)

            for ds_param in pixeloffsets_ds_params:
                add_dataset_and_attrs(pixeloffsets_group, ds_param)
                                                
    def add_parameters_to_procinfo(self):
        """
        Add the parameters group to the "processingInformation" group
        
        Return
        ------
        - params_group (h5py.Group): the parameters group object
        """
        params_group = self.require_group(self.group_paths.ParametersPath)

        self.add_common_to_procinfo_params()
        self.add_RSLC_to_procinfo_params("reference")
        self.add_RSLC_to_procinfo_params("secondary")
        
        runconfig_contents = DatasetParams(
            "runConfigurationContents",
            np.string_(str(self.cfg)),
            np.string_(
                "Contents of the run configuration file with parameters"
                " used for processing"
            ),
        )
        add_dataset_and_attrs(params_group, runconfig_contents)

        return params_group
    
    def add_inputs_to_procinfo(
        self,
        runconfig_path: str,
    ):
        """
        Add the inputs group to the "processingInformation" group
        
        Return
        ------
        - inputs_group (h5py.Group): the inputs group object
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
        inputs_group = self.require_group(f"{self.group_paths.ProcessingInformationPath}/inputs")
        for ds_param in inputs_ds_params:
            add_dataset_and_attrs(inputs_group, ds_param)

        return inputs_group
    
    def _copy_group_by_name(
        self,
        parent_group: h5py.Group,
        src_group_name: str,
        dst_group: h5py.Group,
        dst_group_name: Optional[str] = None,
    ):
        """
        Copy the group name under the parent group to destinated group.
        This function is to handle the case when there is no group name
        under the parent group, but need to create a new group

        Parameters:
        ------
        - parent_group (h5py.Group): the parent group of the src_group_name
        - src_group_name (str): the group name under the  parent group
        - dst_group (h5py.Group): the destinated group
        - dst_group_name (Optional[str]): the new group name, if it is None,
                                          it will use the src group name
        """
        try:
            parent_group.copy(src_group_name, dst_group)
        except KeyError:
            # create a new group 
            if dst_group_name is None:
                dst_group.require_group(src_group_name)
            else:
                dst_group.require_group(dst_group_name)

    def _copy_dataset_by_name(
        self,
        parent_group: h5py.Group,
        src_dataset_name: str,
        dst_group: h5py.Group,
        dst_dataset_name: Optional[str] = None,
    ):
        """
        Copy the dataset under the parent group to destinated group.
        This function is to handle the case when there is no dataset name
        under the parent group, but need to create a new dataset

        Parameters:
        ------
        - parent_group (h5py.Group): the parent group of the src_group_name
        - src_dataset_name (str): the dataset name under the  parent group
        - dst_group (h5py.Group): the destinated group
        - dst_dataset_name (Optional[str]): the new dataset name, if it is None,
                                            it will use the src dataset name
        """
        try:
            parent_group.copy(src_dataset_name, dst_group, dst_dataset_name)
        except KeyError:
            # create a new dataset with the value "NotFoundFromRSLC"
            if dst_dataset_name is None:
                dst_group.create_dataset(
                    src_dataset_name, data=np.string_("NotFoundFromRSLC")
                )
            else:
                dst_group.create_dataset(
                    dst_dataset_name, data=np.string_("NotFoundFromRSLC")
                )

    def _is_geocoded_product(self):
        """
        is Geocoded product
        """
        return self.product_info.isGeocoded
    
    def _get_product_level(self):
        """
        Get the product level.
        """
        return self.product_info.ProductLevel
    
    def _get_product_version(self):
        """
        Get the product version.
        """
        return self.product_info.ProductVersion
    
    def _get_product_type(self):
        """
        Get the product type.
        """
        return self.product_info.ProductType
    
    def _get_product_specification(self):
        """
        Get the product specification
        """
        return self.product_info.ProductSpecificationVersion
    
    def _get_metadata_path(self):
        """
        Get the InSAR product metadata path.
        """
        return self.group_paths.MetadataPath

    def add_common_metadata_to_hdf5(self):
        """
        Write metadata datasets and attributes common to all InSAR products to HDF5
        """

        # Can copy entirety of attitude
        ref_metadata_group = self.ref_h5py_file_obj[self.ref_rslc.MetadataPath]
        dst_metadata_group = self.require_group(self._get_metadata_path())
        self._copy_group_by_name(
            ref_metadata_group, "attitude", dst_metadata_group
        )

        # Orbit population based in inputs
        if self.external_orbit_path is None:
            self._copy_group_by_name(
                ref_metadata_group, "orbit", dst_metadata_group
            )
        else:
            # populate orbit group with contents of external orbit file
            orbit = load_orbit_from_xml(self.external_orbit_path, self.epoch)
            orbit_group = dst_metadata_group.require_group("orbit")
            orbit.save_to_h5(orbit_group)

    def add_identification_to_hdf5(self):
        """
        Add the identification group to the product
        
        Return 
        ------
        dst_id_group (h5py.Group): identification group object
        """

        radar_band_name = self._get_band_name()
        processing_center = self.cfg["primary_executable"].get(
            "processing_center"
        )
        processing_type = self.cfg["primary_executable"].get("processing_type")

        # processing center (JPL or NRSA)
        if processing_center is None:
            processing_center = "JPL"
        elif processing_center.upper() == "J":
            processing_center = "JPL"
        else:
            processing_center = "NRSA"

        # Determine processing type and from it urgent observation
        if processing_type is None:
            processing_type = "UNDEFINED"
        elif processing_type.upper() == "PR":
            processing_type = "NOMINAL"
        elif processing_type.upper() == "UR":
            processing_type = "URGENT"
        else:
            processing_type = "UNDEFINED"
        is_urgent_observation = True if processing_type == "URGENT" else False

        # Extract relevant identification from reference RSLC
        ref_id_group = self.ref_h5py_file_obj[self.ref_rslc.IdentificationPath]
        sec_id_group = self.sec_h5py_file_obj[self.sec_rslc.IdentificationPath]
        dst_id_group = self.require_group(self.group_paths.IdentificationPath)

        # Datasets that need to be copied from the RSLC
        ds_names_need_to_copy = [
            "absoluteOrbitNumber",
            "boundingPolygon",
            "diagnosticModeFlag",
            "frameNumber",
            "trackNumer",
            "isDithered",
            "lookDirection",
            "missionId",
            "orbitPassDirection",
            "plannedDatatakeId",
            "plannedObservationId",
        ]
        for ds_name in ds_names_need_to_copy:
            self._copy_dataset_by_name(ref_id_group, ds_name, dst_id_group)

        # Copy the zeroDopper information from both reference and secondary RSLC
        for ds_name in ["zeroDopplerStartTime", "zeroDopplerEndTime"]:
            self._copy_dataset_by_name(
                ref_id_group, ds_name, dst_id_group, f"referenceZ{ds_name[1:]}"
            )
            self._copy_dataset_by_name(
                sec_id_group, ds_name, dst_id_group, f"secodnaryZ{ds_name[1:]}"
            )

        ds_params = [
            DatasetParams(
                "instrumentName",
                f"{radar_band_name}SAR",
                (
                    "Name of the instrument used to collect the remote"
                    " sensing data provided in this product"
                ),
            ),
            self._get_mixed_mode(),
            DatasetParams(
                "isUrgentObservation",
                is_urgent_observation,
                "Boolean indicating if observation is nominal or urgent",
            ),
            DatasetParams(
                "listOfFrequencies",
                list(self.freq_pols),
                "List of frequency layers available in the product",
            ),
            DatasetParams(
                "processingCenter", processing_center, "Data processing center"
            ),
            DatasetParams(
                "processingDateTime",
                datetime.utcnow().replace(microsecond=0).isoformat(),
                (
                    "Processing UTC date and time in the format"
                    " YYYY-MM-DDTHH:MM:SS"
                ),
            ),
            DatasetParams(
                "processingType",
                processing_type,
                "NOMINAL (or) URGENT (or) CUSTOM (or) UNDEFINED",
            ),
            DatasetParams(
                "radarBand", radar_band_name, "Acquired frequency band"
            ),
            DatasetParams(
                "productLevel",
                self._get_product_level(),
                (
                    "Product level. L0A: Unprocessed instrument data; L0B:"
                    " Reformatted,unprocessed instrument data; L1: Processed"
                    " instrument data in radar coordinates system; and L2:"
                    " Processed instrument data in geocoded coordinates system"
                ),
            ),
            DatasetParams(
                "productVersion",
                self._get_product_version(),
                (
                    "Product version which represents the structure of the"
                    " product and the science content governed by the"
                    " algorithm, input data, and processing parameters"
                ),
            ),
            DatasetParams("productType", self._get_product_type(), "Product type"),
            DatasetParams(
                "productSpecificationVersion",
                self._get_product_specification(),
                (
                    "Product specification version which represents the schema"
                    " of this product"
                ),
            ),
            DatasetParams(
                "isGeocoded",
                np.bool_(self._is_geocoded_product()),
                "Flag to indicate radar geometry or geocoded product",
            ),
        ]
        for ds_param in ds_params:
            add_dataset_and_attrs(dst_id_group, ds_param)

        return dst_id_group
    
    def _get_band_name(self):
        """
        Get the band name ('L' or 'S')

        Returns:
        ------
        'L' or 'S' (str)
        """
        freq = "A" if "A" in self.freq_pols else "B"
        swath_frequency_path = f"{self.ref_rslc.SwathPath}/frequency{freq}/"
        freq_group = self.ref_h5py_file_obj[swath_frequency_path]
        center_freqency = freq_group["processedCenterFrequency"][()] / 1e9
        # L band 
        if (center_freqency >= 1.0) and (center_freqency <= 2.0):
            return "L"
        else:
            return "S"

    def _get_mixed_mode(self):
        """
        Get the mixed mode
        
        Returns:
        ------
        isMixedMode (DatasetParams)
        """

        mixed_mode = False
        for freq, _, _ in get_cfg_freq_pols(self.cfg):
            swath_frequency_path = (
                f"{self.ref_rslc.SwathPath}/frequency{freq}/"
            )
            ref_swath_frequency_group = self.ref_h5py_file_obj[
                swath_frequency_path
            ]
            sec_swath_frequency_group = self.sec_h5py_file_obj[
                swath_frequency_path
            ]

            # range bandwidth of the reference and secondary RSLC
            ref_range_bandwidth = ref_swath_frequency_group[
                "processedRangeBandwidth"
            ][()]
            sec_range_bandwidth = sec_swath_frequency_group[
                "processedRangeBandwidth"
            ][()]

            # if the reference and secondary RSLC have different range bandwidth,
            # it is in mixed mode (i.e. mixed mode = True)
            if abs(ref_range_bandwidth - sec_range_bandwidth) >= 1:
                mixed_mode = True
                
            return DatasetParams(
                "isMixedMode",
                np.bool_(mixed_mode),
                np.string_(
                    '"True" if this product is a composite of data'
                    ' collected in multiple radar modes, "False"'
                    " otherwise."
                ),
            )

    def _get_default_chunks(self):
        """
        Get the defualt chunk size.
        To change the chunks of the children classes, need to overwrite this function
        
        Returns:
        ------
        (128, 128) (tuble) 
        """
        return (128, 128)

    def _create_2d_dataset(
        self,
        h5_group: h5py.Group,
        name: str,
        shape: tuple,
        dtype: object,
        description: str,
        units: Optional[str] = None,
        grid_mapping: Optional[str] = None,
        standard_name: Optional[str] = None,
        long_name: Optional[str] = None,
        yds: Optional[h5py.Dataset] = None,
        xds: Optional[h5py.Dataset] = None,
        fill_value: Optional[Any] = None,
    ):
        """
        Create an empty 2 dimensional dataset under the h5py.Group

        Parameters:
        ------
        - h5_group (h5py.Group): the parent HDF5 group where the new dataset will be stored
        - name (str): dataset name
        - shape (tuple): shape of the dataset
        - dtype (object): data type of the dataset
        - description (str): description of the dataset
        - units (Optional[str]): units of the dataset
        - grid_mapping (Optional[str]): grid mapping string, e.g. "projection"
        - standard_name (Optional[str]): standard name
        - long_name (Optional[str]): long name
        - yds (Optional[h5py.Dataset]): y coordinates
        - xds (Optional[h5py.Dataset]): x coordinates
        - fill_value (Optional[Any]): novalue of the dataset
        """

        # use the default chunk size if the chunk_size is None
        chunks = self._get_default_chunks()

        # do not create chunked dataset if any chunk dimension is larger than
        # dataset or is GUNW (temporary fix for CUDA geocode insar's inability
        # to direct write to HDF5 with chunks)
        # details https://github-fn.jpl.nasa.gov/isce-3/isce/issues/813
        create_with_chunks = (
            chunks[0] < shape[0] and chunks[1] < shape[1]
        ) and ("GUNW" not in h5_group.name)
        if create_with_chunks:
            ds = h5_group.require_dataset(
                name, dtype=dtype, shape=shape, chunks=chunks
            )
        else:
            # create dataset without chunks
            ds = h5_group.require_dataset(name, dtype=dtype, shape=shape)

        # set attributes
        ds.attrs["description"] = np.string_(description)

        if units is not None:
            ds.attrs["units"] = np.string_(units)

        if grid_mapping is not None:
            ds.attrs["grid_mapping"] = np.string_(grid_mapping)

        if standard_name is not None:
            ds.attrs["standard_name"] = np.string_(standard_name)

        if long_name is not None:
            ds.attrs["long_name"] = np.string_(long_name)

        if yds is not None:
            ds.dims[0].attach_scale(yds)

        if xds is not None:
            ds.dims[1].attach_scale(xds)

        if fill_value is not None:
            ds.attrs.create("_FillValue", data=fill_value)
        # create fill value if not speficied
        elif np.issubdtype(dtype, np.floating):
            ds.attrs.create("_FillValue", data=np.nan)
        elif np.issubdtype(dtype, np.integer):
            ds.attrs.create("_FillValue", data=255)
        elif np.issubdtype(dtype, np.complexfloating):
            ds.attrs.create("_FillValue", data=np.nan + 1j * np.nan)