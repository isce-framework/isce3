import h5py
import numpy as np
from nisar.workflows.h5_prep import get_off_params
from nisar.workflows.helpers import get_cfg_freq_pols

from .common import InSARProductsInfo
from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_L1_writer import L1InSARWriter
from .product_paths import RIFGGroupsPaths


class RIFGWriter(L1InSARWriter):
    """
    Writer class for RIFG product inherenting from L1InSARWriter
    """

    def __init__(self, **kwds):
        """
        Constructor for RIFG class
        """
        super().__init__(**kwds)

        # RIFG group paths
        self.group_paths = RIFGGroupsPaths()

        # RIFG product information
        self.product_info = InSARProductsInfo.RIFG()

    def add_root_attrs(self):
        """
        add root attributes
        """

        super().add_root_attrs()

        # Add additional attributes
        self.attrs["title"] = np.string_("NISAR L1_RIFG Product")
        self.attrs["reference_document"] = np.string_("JPL-102270")

        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(self["/"].id, np.string_("complex64"))
        
    def add_algorithms_to_procinfo(self):
        """
        Add the algorithms to processingInformation group

        Returns
        ----------
        algo_group : h5py.Group)
            the algorithm group object
        """
        
        algo_group = super().add_algorithms_to_procinfo()
        self.add_interferogramformation_to_algo(algo_group)
        
        return algo_group
    
    def add_swaths_to_hdf5(self):
        """
        Add swaths to the HDF5
        """
        proc_cfg = self.cfg["processing"]
        rg_looks = proc_cfg["crossmul"]["range_looks"]
        az_looks = proc_cfg["crossmul"]["azimuth_looks"]

        # pull the offset parameters
        is_roff = proc_cfg["offsets_product"]["enabled"]
        margin = get_off_params(proc_cfg, "margin", is_roff)
        rg_gross = get_off_params(proc_cfg, "gross_offset_range", is_roff)
        az_gross = get_off_params(proc_cfg, "gross_offset_azimuth", is_roff)
        rg_start = get_off_params(proc_cfg, "start_pixel_range", is_roff)
        az_start = get_off_params(proc_cfg, "start_pixel_azimuth", is_roff)
        rg_skip = get_off_params(proc_cfg, "skip_range", is_roff)
        az_skip = get_off_params(proc_cfg, "skip_azimuth", is_roff)
        rg_search = get_off_params(
            proc_cfg,
            "half_search_range",
            is_roff,
            pattern="layer",
            get_min=True,
        )
        az_search = get_off_params(
            proc_cfg,
            "half_search_azimuth",
            is_roff,
            pattern="layer",
            get_min=True,
        )
        rg_chip = \
            get_off_params(proc_cfg, "window_range", is_roff, pattern="layer",
                           get_min=True)

        az_chip = \
            get_off_params(proc_cfg, "window_azimuth", is_roff,
                           pattern="layer", get_min=True)

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
            rslc_swaths_group = \
                self.ref_h5py_file_obj[f"{self.ref_rslc.SwathPath}"]

            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"
            ]

            list_of_pols = DatasetParams(
                "listOfPolarizations",
                np.string_(pol_list),
                f"List of processed polarization layers with frequency{freq}",
            )
            add_dataset_and_attrs(swaths_freq_group, list_of_pols)

            # get the RSLC lines and columns
            slc_dset = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}/{pol_list[0]}"
            ]
            slc_lines, slc_cols = slc_dset.shape

            off_length = get_off_params(proc_cfg, "offset_length", is_roff)
            off_width = get_off_params(proc_cfg, "offset_width", is_roff)
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

            rslc_freq_group.copy(
                "processedCenterFrequency",
                swaths_freq_group,
                "centerFrequency",
            )
            rslc_freq_group.copy("numberOfSubSwaths", swaths_freq_group)

            scene_center_params = [
                DatasetParams(
                    "sceneCenterAlongTrackSpacing",
                    rslc_freq_group["sceneCenterAlongTrackSpacing"][()]
                    * az_looks,
                    (
                        "Nominal along track spacing in meters between"
                        " consecutive lines near mid swath of the RIFG image"
                    ),
                    {"units": "meters"},
                ),
                DatasetParams(
                    "sceneCenterGroundRangeSpacing",
                    rslc_freq_group["sceneCenterGroundRangeSpacing"][()]
                    * rg_looks,
                    (
                        "Nominal ground range spacing in meters between"
                        " consecutive pixels near mid swath of the RIFG image"
                    ),
                    {"units": "meters"},
                ),
            ]
            for ds_param in scene_center_params:
                add_dataset_and_attrs(swaths_freq_group, ds_param)

            # valid samples subswath
            num_of_subswaths = rslc_freq_group["numberOfSubSwaths"][()]
            for sub in range(num_of_subswaths):
                subswath = sub + 1
                # Get RSLC subswath dataset, range looks, and destination
                # dataset name based on keys in RSLC
                valid_samples_subswath_name = f"validSamplesSubSwath{subswath}"
                if valid_samples_subswath_name in rslc_freq_group.keys():
                    rslc_freq_subswath_ds = \
                        rslc_freq_group[valid_samples_subswath_name]
                    number_of_range_looks =rslc_freq_subswath_ds[()] \
                            // rg_looks
                else:
                    rslc_freq_subswath_ds = rslc_freq_group["validSamples"]
                    number_of_range_looks = rslc_freq_subswath_ds[()] // rg_looks
                    valid_samples_subswath_name = "validSamples"

                # Create subswath dataset and update attributes from RSLC
                dst_subswath_ds = swaths_freq_group.require_dataset(
                    name=valid_samples_subswath_name,
                    data=number_of_range_looks,
                    shape=number_of_range_looks.shape,
                    dtype=number_of_range_looks.dtype,
                )
                dst_subswath_ds.attrs.update(rslc_freq_subswath_ds.attrs)

            # add the slantRange, zeroDopplerTime, and their spacings to pixel offset group
            offset_slant_range = \
                rslc_freq_group["slantRange"][()][rg_start::rg_skip][:off_width]
            offset_zero_doppler_time = \
                rslc_swaths_group["zeroDopplerTime"][()][az_start::az_skip][:off_length]
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

            def max_look_idx(max_val, n_looks):
                # internal convenience function to get max multilooked index value
                return (
                    np.arange((len(max_val) // n_looks) * n_looks)[::n_looks]
                    + n_looks // 2
                )

            rg_idx, az_idx = (
                max_look_idx(max_val, n_looks)
                for max_val, n_looks in (
                    (igram_slant_range, rg_looks),
                    (igram_zero_doppler_time, az_looks),
                )
            )

            igram_slant_range = igram_slant_range[rg_idx]
            igram_zero_doppler_time = igram_zero_doppler_time[az_idx]
            igram_zero_doppler_time_spacing = \
                rslc_swaths_group["zeroDopplerTimeSpacing"][()] * az_looks
            igram_slant_range_spacing = \
                rslc_freq_group["slantRangeSpacing"][()] * rg_looks

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
                igram_pol_group_name = (
                    f"{swaths_freq_group_name}/interferogram/{pol}"
                )
                offset_pol_group_name = (
                    f"{swaths_freq_group_name}/pixelOffsets/{pol}"
                )

                igram_pol_group = self.require_group(igram_pol_group_name)
                offset_pol_group = self.require_group(offset_pol_group_name)

                # Interferogram datasets
                igram_ds_params = [
                    (
                        "coherenceMagnitude",
                        np.float32,
                        f"Coherence magnitude between {pol} layers",
                        "unitless",
                    ),
                    (
                        "wrappedInterferogram",
                        np.complex64,
                        f"Interferogram between {pol} layers",
                        "DN",
                    ),
                ]

                for igram_ds_param in igram_ds_params:
                    ds_name, ds_dtype, ds_description, ds_unit = igram_ds_param
                    self._create_2d_dataset(
                        igram_pol_group,
                        ds_name,
                        igram_shape,
                        ds_dtype,
                        ds_description,
                        units=ds_unit,
                    )

                # pixelOffsets datasets
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
