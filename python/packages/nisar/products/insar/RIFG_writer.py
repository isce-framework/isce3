import h5py
import numpy as np
from nisar.workflows.helpers import get_cfg_freq_pols

from .InSAR_L1_writer import L1InSARWriter
from .InSAR_products_info import InSARProductsInfo
from .product_paths import RIFGGroupsPaths
from .units import Units


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

    def add_parameters_to_procinfo_group(self):
        """
        Add the parameters group to the "processingInformation" group
        """
        super().add_parameters_to_procinfo_group()

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
                        Units.unitless,
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
