
from datetime import datetime
from typing import Any, Optional

import h5py
import numpy as np
from isce3.core import DateTime
from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows.helpers import get_cfg_freq_pols

from .common import get_validated_file_path
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

        self.ref_h5py_file_obj = h5py.File(
            self.ref_h5_slc_file, "r", libver="latest", swmr=True
        )
        self.sec_h5py_file_obj = h5py.File(
            self.sec_h5_slc_file, "r", libver="latest", swmr=True
        )

    def add_root_attrs(self):
        """
        Write attributes to root that are common to all InSAR products
        """
        self.attrs["Conventions"] = np.string_("CF-1.7")
        self.attrs["contact"] = np.string_("nisarops@jpl.nasa.gov")
        self.attrs["institution"] = np.string_("NASA JPL")
        self.attrs["mission_name"] = np.string_("NISAR")

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

    def _get_metadata_path(self):
        """
        Get the InSAR product metadata path.
        To change the metadata path of the children classes, need to overwrite this function.
        """
        
        return ""

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

    def add_identification_group(self):
        """
        Add the identification group to the product
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
        dst_id_group = self.require_group(CommonPaths.IdentificationPath)

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
        ]
        for ds_param in ds_params:
            add_dataset_and_attrs(dst_id_group, ds_param)

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