import h5py
import numpy as np
from nisar.workflows.h5_prep import get_off_params
from nisar.workflows.helpers import get_cfg_freq_pols

from .common import InSARProductsInfo
from .dataset_params import DatasetParams, add_dataset_and_attrs
from .product_paths import RIFGGroupsPaths
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
        
        # group paths are RIFG group paths
        self.group_paths = RIFGGroupsPaths()
        
        # RIFG product information
        self.product_info = InSARProductsInfo.RIFG()
        
    def add_root_attrs(self):
        """
        add root attributes
        """
        
        super().add_root_attrs()
        
        # Add additional attributes
        self.attrs["title"] = "NISAR L1 RIFG Product"
        self.attrs["reference_document"] = "TBD"
        
        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(self["/"].id, np.string_("complex64"))

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
                f"{self.group_paths.SwathsPath}/frequency{freq}"
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
