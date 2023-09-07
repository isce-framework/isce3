import h5py
import numpy as np
from nisar.workflows.helpers import get_cfg_freq_pols

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_L1_writer import L1InSARWriter
from .InSAR_products_info import InSARProductsInfo
from .product_paths import RIFGGroupsPaths


class RIFGWriter(L1InSARWriter):
    """
    Writer class for RIFG product inherenting from L1InSARWriter
    """
    def __init__(self, **kwds):
        """
        Constructor for RIFG class with additional range and azimuth looks
        variables for the interferogram multilooking
        """
        super().__init__(**kwds)

        # RIFG group paths
        self.group_paths = RIFGGroupsPaths()

        # RIFG product information
        self.product_info = InSARProductsInfo.RIFG()

        # Azimuth and Range Looks
        proc_cfg = self.cfg["processing"]
        self.igram_range_looks = proc_cfg["crossmul"]["range_looks"]
        self.igram_azimuth_looks = proc_cfg["crossmul"]["azimuth_looks"]

    def add_root_attrs(self):
        """
        Add root attributes
        """
        super().add_root_attrs()

        # Add additional attributes
        self.attrs["title"] = np.string_("NISAR L1 RIFG Product")
        self.attrs["reference_document"] = \
            np.string_("D-102270 NISAR NASA SDS Product Specification"
                       " L1 Range Doppler Wrapped Interferogram")

        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(self["/"].id, np.string_("complex64"))

    def add_interferogram_to_procinfo_params_group(self):
        """
        Add the interferogram group to "processingInformation/parameters group"
        """
        proc_cfg_crossmul = self.cfg["processing"]["crossmul"]
        range_filter = proc_cfg_crossmul["common_band_range_filter"]
        azimuth_filter = proc_cfg_crossmul["common_band_azimuth_filter"]

        flatten = proc_cfg_crossmul["flatten"]

        interferogram_ds_params = [
            DatasetParams(
                "commonBandRangeFilterApplied",
                np.bool_(range_filter),
                (
                    "Flag to indicate if common band range filter has been"
                    " applied"
                ),
            ),
            DatasetParams(
                "commonBandAzimuthFilterApplied",
                np.bool_(azimuth_filter),
                (
                    "Flag to indicate if common band azimuth filter has been"
                    " applied"
                ),
            ),
            DatasetParams(
                "ellipsoidalFlatteningApplied",
                np.bool_(flatten),
                (
                    "Flag to indicate if interferometric phase has been"
                    " flattened with respect to a zero height ellipsoid"
                ),
            ),
            DatasetParams(
                "topographicFlatteningApplied",
                np.bool_(flatten),
                (
                    "Flag to indicate if interferometric phase has been"
                    " flattened with respect to a zero height ellipsoid"
                ),
            ),
            DatasetParams(
                "numberOfRangeLooks",
                np.uint32(self.igram_range_looks),
                (
                    "Number of looks applied in the slant range direction to"
                    " form the wrapped interferogram"
                ),
                {
                    "units": "unitless",
                },
            ),
            DatasetParams(
                "numberOfAzimuthLooks",
                np.uint32(self.igram_azimuth_looks),
                (
                    "Number of looks applied in the along-track direction to"
                    " form the wrapped interferogram"
                ),
                {
                    "units": "unitless",
                },
            ),
        ]

        for freq, *_ in get_cfg_freq_pols(self.cfg):
            bandwidth_group_path = f"{self.ref_rslc.SwathPath}/frequency{freq}"
            bandwidth_group = self.ref_h5py_file_obj[bandwidth_group_path]

            igram_group_name = \
                f"{self.group_paths.ParametersPath}/wrappedInterferogram/frequency{freq}"
            igram_group = self.require_group(igram_group_name)

            # TODO: the azimuthBandwidth and rangeBandwidth are placeholders heres,
            # and copied from the bandpassed RSLC data.
            # those should be updated in the crossmul module.
            bandwidth_group.copy(
                "processedAzimuthBandwidth",
                igram_group,
                "azimuthBandwidth",
            )
            bandwidth_group.copy(
                "processedRangeBandwidth",
                igram_group,
                "rangeBandwidth",
            )

            for ds_param in interferogram_ds_params:
                add_dataset_and_attrs(igram_group, ds_param)

    def add_parameters_to_procinfo_group(self):
        """
        Add the parameters group to the "processingInformation" group
        """
        super().add_parameters_to_procinfo_group()
        self.add_interferogram_to_procinfo_params_group()

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithms to processingInformation group
        """
        super().add_algorithms_to_procinfo_group()
        self.add_interferogramformation_to_algo_group()

    def add_interferogram_to_swaths_group(self):
        """
        Add interferogram group to swaths
        """
        super().add_interferogram_to_swaths_group()

        # Add the wrappedInterferogram to the interferogram group
        # under swaths group
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            swaths_freq_group_name = (
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            )

            # shape of the interferogram product
            igram_shape = self._get_interferogram_dataset_shape(freq, pol_list[0])

            # add the wrappedInterferogram to each polarization group
            for pol in pol_list:
                # create the interferogram dataset
                igram_pol_group_name = \
                    f"{swaths_freq_group_name}/interferogram/{pol}"
                igram_pol_group = self.require_group(igram_pol_group_name)

                # The interferogram dataset parameters including the
                # dataset name, dataset data type, description, units
                igram_ds_params = [
                    (
                        "wrappedInterferogram",
                        np.complex64,
                        f"Interferogram between {pol} layers",
                        "DN",
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
                    )

    def add_swaths_to_hdf5(self):
        """
        Add swaths to the HDF5
        """
        super().add_swaths_to_hdf5()

        # add subswaths to swaths group
        self.add_subswaths_to_swaths_group()
        self.add_interferogram_to_swaths_group()
