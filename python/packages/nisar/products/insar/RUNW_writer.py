import numpy as np
from nisar.workflows.helpers import get_cfg_freq_pols

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_L1_writer import L1InSARWriter
from .InSAR_products_info import InSARProductsInfo
from .product_paths import RUNWGroupsPaths
from .units import Units


class RUNWWriter(L1InSARWriter):
    """
    Writer class for RUNW product inherent from L1InSARWriter
    """
    def __init__(self, **kwds):
        """
        Constructor for RUNW class with additional range and azimuth
        looks variables for the phase unwrapping
        """
        super().__init__(**kwds)

        # group paths are RUNW group paths
        self.group_paths = RUNWGroupsPaths()

        # RUNW product information
        self.product_info = InSARProductsInfo.RUNW()

        proc_cfg = self.cfg["processing"]
        self.igram_range_looks = proc_cfg["crossmul"]["range_looks"]
        self.igram_azimuth_looks = proc_cfg["crossmul"]["azimuth_looks"]
        unwrap_rg_looks = proc_cfg["phase_unwrap"]["range_looks"]
        unwrap_az_looks = proc_cfg["phase_unwrap"]["azimuth_looks"]

        # replace the looks from the unwrap looks when
        # unwrap_az_looks !=1 or unwrap_rg_looks != 1, i.e.,
        # when the both unwrap_az_looks and unwrap_rg_looks are euqals to 1
        # the rg and az looks from the crossmul will be applied.
        # NOTE: unwrap looks here are the total looks on the RSLC, not on top of the RIFG
        if (unwrap_az_looks != 1) or (unwrap_rg_looks != 1):
            self.igram_range_looks = unwrap_rg_looks
            self.igram_azimuth_looks = unwrap_az_looks


    def add_root_attrs(self):
        """
        add root attributes
        """
        super().add_root_attrs()

        self.attrs["title"] = np.string_("NISAR L1 RUNW Product")
        self.attrs["reference_document"] = \
            np.string_("D-102271 NISAR NASA SDS Product Specification"
                       " L1 Range Doppler UnWrapped Interferogram")

    def add_ionosphere_to_procinfo_params_group(self):
        """
        Add the ionosphere to the processingInformation/parameters group
        """
        high_bandwidth = 0
        low_bandwidth = 0
        iono_cfg = self.cfg["processing"]["ionosphere_phase_correction"]

        if iono_cfg["enabled"]:
            high_bandwidth = iono_cfg["split_range_spectrum"]\
                ["high_band_bandwidth"]
            low_bandwidth = iono_cfg["split_range_spectrum"]\
                ["low_band_bandwidth"]

        ds_params = [
            DatasetParams(
                "highBandBandwidth",
                np.float64(high_bandwidth),
                "Slant range bandwidth of the high sub-band image",
                {
                    "units": Units.hertz,
                },
            ),
            DatasetParams(
                "lowBandBandwidth",
                np.float64(low_bandwidth),
                "Slant range bandwidth of the low sub-band image",
                {
                    "units": Units.hertz,
                },
            ),
        ]

        iono_group = self.require_group(
            f"{self.group_paths.ParametersPath}/ionosphere"
        )
        for ds_param in ds_params:
            add_dataset_and_attrs(iono_group, ds_param)

    def add_ionosphere_estimation_to_algo_group(self):
        """
        Add the ionosphere estimation group to algorithms group
        """
        iono_algorithm = "None"
        iono_filling = "None"
        iono_filtering = "None"
        iono_outliers = "None"
        unwrap_correction = False
        num_of_iters = 0
        iono_cfg = self.cfg["processing"]["ionosphere_phase_correction"]

        if iono_cfg["enabled"]:
            iono_algorithm = iono_cfg["spectral_diversity"]

            # ionosphere filling method
            iono_filling = iono_cfg["dispersive_filter"]["filling_method"]
            num_of_iters = iono_cfg["dispersive_filter"]["filter_iterations"]
            # ionosphere filtering method
            iono_filtering = f"Iterative gaussian filter with {num_of_iters} filtering"
            iono_outliers = iono_cfg["dispersive_filter"]["filter_mask_type"]
            unwrap_correction = iono_cfg["dispersive_filter"][
                "unwrap_correction"
            ]

        if iono_algorithm == "split_main_band":
            iono_algorithm = (
                "Range split-spectrum with sub-band sub-images obtained by"
                " splitting the main range bandwidth of the input RSLCs"
            )
        elif iono_algorithm == "main_side_band":
            iono_algorithm = (
                "Range split-spectrum with sub-band images being the main band"
                " and the side band of the input RSLCs"
            )
        elif iono_algorithm == "main_diff_ms_band":
            iono_algorithm = (
                "Range split-spectrum with sub-band interferograms from the"
                " main band and the difference of the main and side band of"
                " the input RSLCs"
            )

        ds_params = [
            DatasetParams(
                "ionosphereAlgorithm",
                iono_algorithm,
                "Algorithm used to estimate ionosphere phase screen",
                {
                    "algorithm_type": "Ionosphere estimation",
                },
            ),
            DatasetParams(
                "ionosphereFilling",
                iono_filling,
                "Outliers data filling algorithm"
                " for ionosphere phase estimation"
                ,
                {
                    "algorithm_type": "Ionosphere estimation",
                },
            ),
            DatasetParams(
                "ionosphereFiltering",
                iono_filtering,
                f"Filtering algorithm for ionosphere phase screen computation",
                {
                    "algorithm_type": "Ionosphere estimation",
                },
            ),
            DatasetParams(
                "ionosphereOutliers",
                iono_outliers,
                "Algorithm identifying outliers in unfiltered ionosphere"
                " phase screen"
                ,
                {
                    "algorithm_type": "Ionosphere estimation",
                },
            ),
            DatasetParams(
                "unwrappingErrorCorrection",
                np.bool_(unwrap_correction),
                "Algorithm correcting unwrapping errors in sub-band"
                " unwrapped interferograms"
                ,
                {
                    "algorithm_type": "Ionosphere estimation",
                },
            ),
        ]

        iono_est_group = self.require_group(
            f"{self.group_paths.AlgorithmsPath}/ionosphereEstimation")
        for ds_param in ds_params:
            add_dataset_and_attrs(iono_est_group, ds_param)

    def add_unwarpping_to_algo_group(self):
        """
        Add the unwrapping to the algorithms group
        """
        cost_mode = "None"
        unwrapping_algorithm = "None"
        unwrapping_initializer = "None"
        phase_filling = "None"
        phase_outliers = "None"

        unwrap_cfg = self.cfg["processing"]["phase_unwrap"]
        unwrapping_algorithm = unwrap_cfg["algorithm"]
        prep_unwrap_cfg = unwrap_cfg["preprocess_wrapped_phase"]

        if unwrapping_algorithm.lower() == "snaphu":
            cost_mode = unwrap_cfg["snaphu"]["cost_mode"]
            unwrapping_initializer = unwrap_cfg["snaphu"][
                "initialization_method"
            ]
            if cost_mode is None:
                cost_mode = "None"
            if unwrapping_initializer is None:
                unwrapping_initializer = "None"

        if prep_unwrap_cfg["enabled"]:
            phase_filling = prep_unwrap_cfg["filling_method"]
            phase_outliers = prep_unwrap_cfg["mask"]["mask_type"]

            if phase_filling is None:
                phase_filling = "None"
            if phase_outliers is None:
                phase_outliers = "None"

        ds_params = [
            DatasetParams(
                "costMode",
                cost_mode,
                "Cost mode algorithm for phase unwrapping",
                {
                    "algorithm_type": "Unwrapping",
                },
            ),
            DatasetParams(
                "unwrappingAlgorithm",
                unwrapping_algorithm,
                "Algorithm used for phase unwrapping",
                {
                    "algorithm_type": "Unwrapping",
                },
            ),
            DatasetParams(
                "unwrappingInitializer",
                unwrapping_initializer,
                "Algorithm used to initialize phase unwrapping",
                {
                    "algorithm_type": "Unwrapping",
                },
            ),
        ]

        unwrap_group = self.require_group(
            f"{self.group_paths.AlgorithmsPath}/unwrapping")
        for ds_param in ds_params:
            add_dataset_and_attrs(unwrap_group, ds_param)

        ds_params = [
            DatasetParams(
                "wrappedPhaseFilling",
                phase_filling,
                "Outliers data filling algorithm for phase unwrapping"
                " preprocessing"
                ,
                {
                    "algorithm_type": "Unwrapping",
                },
            ),
            DatasetParams(
                "wrappedPhaseOutliers",
                phase_outliers,
                "Algorithm identifying outliers in the wrapped"
                " interferogram"
                ,
                {
                    "algorithm_type": "Unwrapping",
                },
            ),
        ]
        unwrap_prep_group = self.require_group(
            f"{self.group_paths.AlgorithmsPath}/unwrapping/preprocessing")
        for ds_param in ds_params:
            add_dataset_and_attrs(unwrap_prep_group, ds_param)

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithms to processingInformation group
        """
        super().add_algorithms_to_procinfo_group()

        self.add_interferogramformation_to_algo_group()
        self.add_ionosphere_estimation_to_algo_group()
        self.add_unwarpping_to_algo_group()

    def add_parameters_to_procinfo_group(self):
        """
        Add parameters group to processingInformation/parameters group
        """
        super().add_parameters_to_procinfo_group()
        self.add_ionosphere_to_procinfo_params_group()

    def add_interferogram_to_swaths_group(self):
        """
        Add interferogram group to swaths group
        """
        super().add_interferogram_to_swaths_group()

        # Add the connectedComponents, ionospherePhaseScreen,
        # ionospherePhaseScreenUncertainty, and the
        # unwrappedPhase datasets
        # to the interferogram group under swaths groups
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            swaths_freq_group_name = (
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            )

            # shape of the interferogram product
            igram_shape = self._get_interferogram_dataset_shape(freq, pol_list[0])

            # add additonal datasets to each polarization group
            for pol in pol_list:
                # create the interferogram dataset
                igram_pol_group_name = \
                    f"{swaths_freq_group_name}/interferogram/{pol}"
                igram_pol_group = self.require_group(igram_pol_group_name)

                # The interferogram dataset parameters including the
                # dataset name, dataset data type, description, units,
                igram_ds_params = [
                    (
                        "connectedComponents",
                        np.uint32,
                        f"Connected components for {pol} layer",
                        Units.dn,
                    ),
                    (
                        "ionospherePhaseScreen",
                        np.float32,
                        "Ionosphere phase screen",
                        Units.radian,
                    ),
                    (
                        "ionospherePhaseScreenUncertainty",
                        np.float32,
                        "Uncertainty of the ionosphere phase screen",
                        Units.radian,
                    ),
                    (
                        "unwrappedPhase",
                        np.float32,
                        f"Unwrapped interferogram between {pol} layers",
                        Units.radian,
                    ),
                ]

                for igram_ds_param in igram_ds_params:
                    ds_name, ds_dtype, ds_description, ds_unit \
                        = igram_ds_param
                    self._create_2d_dataset(
                        igram_pol_group,
                        ds_name,
                        igram_shape,
                        ds_dtype,
                        ds_description,
                        units=ds_unit,
                        compression_enabled=self.cfg['output']['compression_enabled'],
                        compression_level=self.cfg['output']['compression_level'],
                        chunk_size=self.cfg['output']['chunk_size'],
                        shuffle_filter=self.cfg['output']['shuffle']
                    )

    def add_swaths_to_hdf5(self):
        """
        Add Swaths to the HDF5
        """
        super().add_swaths_to_hdf5()

        # add subswaths and interferogram to swaths group
        self.add_subswaths_to_swaths_group()
        self.add_interferogram_to_swaths_group()
