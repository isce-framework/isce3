import h5py
import numpy as np
from nisar.workflows.h5_prep import get_off_params
from nisar.workflows.helpers import get_cfg_freq_pols

from .common import InSARProductsInfo
from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_L1_writer import L1InSARWriter
from .product_paths import RUNWGroupsPaths


class RUNWWriter(L1InSARWriter):
    """
    Writer class for RUNW product inherent from L1InSARWriter
    """

    def __init__(self,**kwds):
        """
        Constructor for RUNW class
        """
        super().__init__(**kwds)

        # group paths are RUNW group paths
        self.group_paths = RUNWGroupsPaths()

        # RUNW product information
        self.product_info = InSARProductsInfo.RUNW()

    def add_root_attrs(self):
        """
        add root attributes
        """

        super().add_root_attrs()

        self.attrs["title"] = np.string_("NISAR L1 RUNW Product")
        self.attrs["reference_document"] = np.string_("JPL-102271")

    def add_ionosphere_to_procinfo_params(self):
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
                np.float32(high_bandwidth),
                "Slant range bandwidth of the high sub-band image",
                {
                    "units": "Hz",
                },
            ),
            DatasetParams(
                "lowBandBandwidth",
                np.float32(low_bandwidth),
                "Slant range bandwidth of the low sub-band image",
                {
                    "units": "Hz",
                },
            ),
        ]

        iono_group = self.require_group(
            f"{self.group_paths.ParametersPath}/ionosphere"
        )
        for ds_param in ds_params:
            add_dataset_and_attrs(iono_group, ds_param)

    def add_ionosphere_est_to_algo(self, algo_group: h5py.Group):
        """
        Add the ionosphere estimation group to algorithms group

        Parameters
        ----------
        algo_group : h5py.Group
            algorithms group object
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
            iono_filtering = "gaussian"
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
        else:
            iono_algorithm = "None"

        ds_params = [
            DatasetParams(
                "ionosphereAlgorithm",
                np.string_(iono_algorithm),
                "Algorithm used to estimate ionosphere phase screen",
                {
                    "algorithm_type": "Ionosphere estimation",
                },
            ),
            DatasetParams(
                "ionosphereFilling",
                np.string_(iono_filling),
                "Outliers data filling algorithm"
                " for ionosphere phase estimation"
                ,
                {
                    "algorithm_type": "Ionosphere estimation",
                },
            ),
            DatasetParams(
                "ionosphereFiltering",
                np.string_(iono_filtering),
                f"Iterative gaussian filter with {num_of_iters} filtering"
                " algorithm for ionosphere phase screen computation"
                ,
                {
                    "algorithm_type": "Ionosphere estimation",
                },
            ),
            DatasetParams(
                "ionosphereOutliers",
                np.string_(iono_outliers),
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
                "Flag indicating if unwrapping errors in sub-band"
                " unwrapped interferograms are corrected"
                ,
                {
                    "algorithm_type": "Ionosphere estimation",
                },
            ),
        ]

        iono_est_group = self.require_group(
            f"{self.group_paths.AlgorithmsPath}/ionosphereEstimation"
        )
        for ds_param in ds_params:
            add_dataset_and_attrs(iono_est_group, ds_param)

    def add_unwarpping_to_algo(self, algo_group: h5py.Group):
        """
        Add the unwrapping to the algorithms group

        Parameters
        ----------
        algo_group : h5py.Group
            algorithms group object
        """

        cost_mode = "None"
        unwrapping_algorithm = "None"
        unwrapping_initializer = "None"
        phase_filling = "None"
        phase_outliers = "None"

        unwrap_cfg = self.cfg["processing"]["phase_unwrap"]
        unwrapping_algorithm = unwrap_cfg["algorithm"]

        if unwrapping_algorithm.lower() == "snaphu":
            # cost mode
            cost_mode = unwrap_cfg["snaphu"]["cost_mode"]
            # unwrapping initializer
            unwrapping_initializer = unwrap_cfg["snaphu"][
                "initialization_method"
            ]

            # if cost mode and unwrapping initializer are empty
            if cost_mode is None:
                cost_mode = "None"
            if unwrapping_initializer is None:
                unwrapping_initializer = "None"

        if unwrap_cfg["preprocess_wrapped_phase"]["enabled"]:
            # wrapped phase filling
            phase_filling = unwrap_cfg["preprocess_wrapped_phase"][
                "filling_method"
            ]
            # wrapped phase outliers
            phase_outliers = unwrap_cfg["preprocess_wrapped_phase"]["mask"][
                "mask_type"
            ]

            # if phase_filling and  phase_outliers are empty
            if phase_filling is None:
                phase_filling = "None"
            if phase_outliers is None:
                phase_outliers = "None"

        ds_params = [
            DatasetParams(
                "costMode",
                np.string_(cost_mode),
                "Cost mode algorithm for phase unwrapping",
                {
                    "algorithm_type": "Unwrapping",
                },
            ),
            DatasetParams(
                "wrappedPhaseFilling",
                np.string_(phase_filling),
                "Outliers data filling algorithm for phase unwrapping"
                " preprocessing"
                ,
                {
                    "algorithm_type": "Unwrapping",
                },
            ),
            DatasetParams(
                "wrappedPhaseOutliers",
                np.string_(phase_outliers),
                "Algorithm identifying outliers in the wrapped"
                " interferogram"
                ,
                {
                    "algorithm_type": "Unwrapping",
                },
            ),
            DatasetParams(
                "unwrappingAlgorithm",
                np.string_(unwrapping_algorithm),
                "Algorithm used for phase unwrapping",
                {
                    "algorithm_type": "Unwrapping",
                },
            ),
            DatasetParams(
                "unwrappingInitializer",
                np.string_(unwrapping_initializer),
                "Algorithm used to initialize phase unwrapping",
                {
                    "algorithm_type": "Unwrapping",
                },
            ),
        ]

        unwrap_group = algo_group.require_group("unwrapping")
        for ds_param in ds_params:
            add_dataset_and_attrs(unwrap_group, ds_param)

    def add_algorithms_to_procinfo(self):
        """
        Add the algorithms to processingInformation group

        Returns
        ----------
        algo_group : h5py.Group)
            the algorithm group object
        """
        
        algo_group = super().add_algorithms_to_procinfo()
        self.add_ionosphere_est_to_algo(algo_group)
        self.add_unwarpping_to_algo(algo_group)
        
        return algo_group

    def add_parameters_to_procinfo(self):
        """
        Add parameters group to processingInformation/parameters group
        """

        super().add_parameters_to_procinfo()
        self.add_ionosphere_to_procinfo_params()

    def add_swaths_to_hdf5(self):
        """
        Add Swaths to the HDF5
        """
        
        super().add_swaths_to_hdf5()
        
        pcfg = self.cfg["processing"]
        rg_looks = pcfg["crossmul"]["range_looks"]
        az_looks = pcfg["crossmul"]["azimuth_looks"]
        unwrap_rg_looks = pcfg["phase_unwrap"]["range_looks"]
        unwrap_az_looks = pcfg["phase_unwrap"]["azimuth_looks"]
        
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
            swaths_freq_group = self.require_group(swaths_freq_group_name)

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

            # shape of the interferogram product
            igram_shape = (
                (slc_lines // az_looks) // unwrap_az_looks,
                (slc_cols // rg_looks) // unwrap_rg_looks,
            )

            rslc_freq_group.copy("numberOfSubSwaths", swaths_freq_group)

            scence_center_params = [
                DatasetParams(
                    "sceneCenterAlongTrackSpacing",
                    rslc_freq_group["sceneCenterAlongTrackSpacing"][()]
                    * az_looks,
                    "Nominal along track spacing in meters between"
                    " consecutive lines near mid swath of the RIFG image"
                    ,
                    {"units": "meters"},
                ),
                DatasetParams(
                    "sceneCenterGroundRangeSpacing",
                    rslc_freq_group["sceneCenterGroundRangeSpacing"][()]
                    * rg_looks,
                    "Nominal ground range spacing in meters between"
                    " consecutive pixels near mid swath of the RIFG image"
                    ,
                    {"units": "meters"},
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
                        // rg_looks // unwrap_rg_looks
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
                        rslc_freq_group["validSamples"][()] // rg_looks // unwrap_rg_looks
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
            offset_slant_range = rslc_freq_group["slantRange"][()]\
                [rg_start::rg_skip][:off_width]
            offset_zero_doppler_time = rslc_swaths_group["zeroDopplerTime"][()]\
                [az_start::az_skip][:off_length]
            offset_zero_doppler_time_spacing = \
                rslc_swaths_group["zeroDopplerTimeSpacing"][()] * az_skip
            offset_slant_range_spacing = \
                rslc_freq_group["slantRangeSpacing"][()] * rg_skip
           
            ds_offsets_params = [
                DatasetParams(
                    "slantRange",
                    offset_slant_range,
                    "Slant range vector",
                    rslc_freq_group["slantRange"].attrs,
                ),
                DatasetParams(
                    "zeroDopplerTime",
                    offset_zero_doppler_time,
                    "Zero Doppler azimuth time vector",
                    rslc_swaths_group["zeroDopplerTime"].attrs,
                ),
                DatasetParams(
                    "zeroDopplerTimeSpacing",
                    offset_zero_doppler_time_spacing,
                    "Along track spacing of the offset grid",
                    rslc_swaths_group["zeroDopplerTimeSpacing"].attrs,
                ),
                DatasetParams(
                    "slantRangeSpacing",
                    offset_slant_range_spacing,
                    "Slant range spacing of offset grid",
                    rslc_freq_group["slantRangeSpacing"].attrs,
                ),
            ]
            
            offset_group_name = f"{swaths_freq_group_name}/pixelOffsets"
            offset_group = self.require_group(offset_group_name)
            for ds_param in ds_offsets_params:
                add_dataset_and_attrs(offset_group, ds_param)

            #  add the slantRange, zeroDopplerTime, and their spacings to inteferogram group
            igram_slant_range = rslc_freq_group["slantRange"][()]
            igram_zero_doppler_time = rslc_swaths_group["zeroDopplerTime"][()]
            
            def max_look_idx(max_val, n_looks, unwrap_looks):
                # internal convenience function to get max multilooked index value
                return (
                    np.arange((len(max_val) // n_looks // unwrap_looks) \
                        * n_looks * unwrap_looks)[::n_looks*unwrap_looks]
                    + n_looks*unwrap_looks // 2
                )

            rg_idx, az_idx = (
                max_look_idx(max_val, n_looks, unwrap_looks)
                for max_val, n_looks, unwrap_looks in (
                    (igram_slant_range, rg_looks, unwrap_rg_looks),
                    (igram_zero_doppler_time, az_looks, unwrap_az_looks),
                )
            )
            
            igram_slant_range = igram_slant_range[rg_idx]
            igram_zero_doppler_time = igram_zero_doppler_time[az_idx]
            igram_zero_doppler_time_spacing = \
                rslc_swaths_group["zeroDopplerTimeSpacing"][()] * az_looks * unwrap_az_looks
            igram_slant_range_spacing = \
                rslc_freq_group["slantRangeSpacing"][()] * rg_looks * unwrap_rg_looks
                
            ds_igram_params = [
                DatasetParams(
                    "slantRange",
                    igram_slant_range,
                    "Slant range vector",
                    rslc_freq_group["slantRange"].attrs,
                ),
                DatasetParams(
                    "zeroDopplerTime",
                    igram_zero_doppler_time,
                    "Zero Doppler azimuth time vector",
                    rslc_swaths_group["zeroDopplerTime"].attrs,
                ),
                DatasetParams(
                    "zeroDopplerTimeSpacing",
                    igram_zero_doppler_time_spacing,
                    (
                        "Time interval in the along track direction for raster"
                        " layers. This is same as the spacing between"
                        " consecutive entries in the zeroDopplerTime array"
                    ),
                    rslc_swaths_group["zeroDopplerTime"].attrs,
                ),
                DatasetParams(
                    "slantRangeSpacing",
                    igram_slant_range_spacing,
                    (
                        "Slant range spacing of grid. Same as difference"
                        " between consecutive samples in slantRange array"
                    ),
                    rslc_freq_group["slantRangeSpacing"].attrs,
                ),
            ]
            igram_group_name = f"{swaths_freq_group_name}/interferogram"
            igram_group = self.require_group(igram_group_name)
            for ds_param in ds_igram_params:
                add_dataset_and_attrs(igram_group, ds_param)

            # add the polarization
            for pol in pol_list:
                
                # create the interferogram dataset
                igram_pol_group_name = \
                    f"{swaths_freq_group_name}/interferogram/{pol}"
                igram_pol_group = self.require_group(igram_pol_group_name)

                igram_ds_params = [
                    (
                        "connectedComponents",
                        np.uint32,
                        f"Connected components for {pol} layers",
                        "DN",
                        0,
                    ),
                    (
                        "ionospherePhaseScreen",
                        np.float32,
                        "Ionosphere phase screen",
                        "radians",
                        None,
                    ),
                    (
                        "ionospherePhaseScreenUncertainty",
                        np.float32,
                        "Uncertainty of the ionosphere phase screen",
                        "radians",
                        None,
                    ),
                    (
                        "coherenceMagnitude",
                        np.float32,
                        f"Coherence magnitude between {pol} layers",
                        "unitless",
                        None,
                    ),
                    (
                        "unwrappedPhase",
                        np.float32,
                        f"Unwrapped Interferogram between {pol} layers",
                        "radians",
                        None,
                    ),
                ]
                
                for igram_ds_param in igram_ds_params:
                    ds_name, ds_dtype, ds_description, ds_unit, ds_filling_value\
                        = igram_ds_param
                    self._create_2d_dataset(
                        igram_pol_group,
                        ds_name,
                        igram_shape,
                        ds_dtype,
                        ds_description,
                        units=ds_unit,
                        fill_value=ds_filling_value
                    )               
                
                # Create the pixel offsets dataset
                offset_pol_group_name = \
                    f"{swaths_freq_group_name}/pixelOffsets/{pol}"
                offset_pol_group = self.require_group(offset_pol_group_name)
                
                pixel_offsets_ds_params = [
                    (
                        "alongTrackOffset",
                        "Along track offset",
                        "meters",
                    ),
                    (
                        "crossCorrelationPeak",
                        "Normalized cross-correlation surface peak",
                        "unitless",
                    ),
                    (
                        "slantRangeOffset",
                        "Slant range offset",
                        "meters",
                    ),
                ]
                
                for pixel_offsets_ds_param in pixel_offsets_ds_params:
                    ds_name, ds_description, ds_unit = pixel_offsets_ds_param
                    self._create_2d_dataset(
                        offset_pol_group,
                        ds_name,
                        off_shape,
                        np.float32,
                        ds_description,
                        units=ds_unit,
                    ) 
