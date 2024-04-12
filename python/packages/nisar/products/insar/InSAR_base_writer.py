import os
from datetime import datetime
from itertools import product
from typing import Any, List, Optional, Union

import h5py
import numpy as np
from isce3.core.types import complex32, to_complex32
from isce3.product import GeoGridParameters, RadarGridParameters
from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows.h5_prep import get_off_params
from nisar.workflows.helpers import get_cfg_freq_pols

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .granule_id import get_insar_granule_id
from .InSAR_products_info import ISCE3_VERSION, InSARProductsInfo
from .product_paths import CommonPaths
from .units import Units
from .utils import extract_datetime_from_string


class InSARBaseWriter(h5py.File):
    """
    The base class of InSAR product inheriting from h5py.File to avoid passing
    h5py.File parameter

    Attributes
    ----------
    cfg : dict
        Runconfig dictionary
    runconfig_path : str
        Path of the runconfig file
    ref_h5_slc_file : str
        Path of the reference RSLC
    sec_h5_slc_file : str
        Path of the secondary RSLC
    external_ref_orbit_path : str
        Path of the external reference orbit file
    external_sec_orbit_path : str
        Path of the external secondary orbit file
    ref_orbit_epoch : Datetime
        The reference datetime for the orbit
    sec_orbit_epoch : Datetime
        The secondary datetime for the orbit
    ref_orbit : isce3.core.Orbit
        The reference orbit object
    sec_orbit : isce3.core.Orbit
        The secondary orbit object
    ref_rslc : nisar.products.readers.SLC
        nisar.products.readers.SLC object of reference RSLC file
    sec_rslc : nisar.products.readers.SLC
        nisar.products.readers.SLC object of the secondary RSLC file
    ref_h5py_file_obj : h5py.File
        h5py.File object of the reference RSLC
    sec_h5py_file_obj : h5py.File
        h5py.File object of the secondary RSLC
    kwds : dict
        Parameters of the h5py.File
    """
    def __init__(self,
                 runconfig_dict: dict,
                 runconfig_path: str,
                 **kwds):
        """
        Constructor of the InSAR Product Base class. Inheriting from h5py.File
        to avoid passing h5py.File parameter.

        Parameters
        ----------
        runconfig_dict : dict
            Runconfig dictionary
        runconfig_path : str
            Path of the runconfig file
        """
        super().__init__(**kwds)

        self.cfg = runconfig_dict
        self.runconfig_path = runconfig_path

        # Reference and Secondary RSLC files
        self.ref_h5_slc_file = \
            self.cfg["input_file_group"]["reference_rslc_file"]

        self.sec_h5_slc_file = \
            self.cfg["input_file_group"]["secondary_rslc_file"]

        # Pull the frequency and polarizations
        self.freq_pols = \
            self.cfg["processing"]["input_subset"]\
                ["list_of_frequencies"]

        # Group paths
        self.group_paths = CommonPaths()

        # Product information
        self.product_info = InSARProductsInfo.Base()

        # Check if reference and secondary exists as files
        orbit_files = \
            self.cfg["dynamic_ancillary_file_group"]["orbit_files"]

        self.external_ref_orbit_path = orbit_files['reference_orbit_file']
        self.external_sec_orbit_path = orbit_files['secondary_orbit_file']

        self.ref_rslc = SLC(hdf5file=self.ref_h5_slc_file)
        self.sec_rslc = SLC(hdf5file=self.sec_h5_slc_file)

        # Pull the radargrid of reference and secondary RSLC
        freq = "A" if "A" in self.freq_pols else "B"
        ref_radargrid = self.ref_rslc.getRadarGrid(freq)
        sec_radargrid = self.sec_rslc.getRadarGrid(freq)

        self.ref_orbit_epoch = ref_radargrid.ref_epoch
        self.sec_orbit_epoch = sec_radargrid.ref_epoch

        self.ref_h5py_file_obj = \
            h5py.File(self.ref_h5_slc_file, "r", libver="latest", swmr=True)

        self.sec_h5py_file_obj = \
            h5py.File(self.sec_h5_slc_file, "r", libver="latest", swmr=True)

        # Create the orbit object and set their reference epochs
        if self.external_ref_orbit_path is not None:
            self.ref_orbit = load_orbit_from_xml(self.external_ref_orbit_path,
                                                 self.ref_orbit_epoch)
        else:
            self.ref_orbit = self.ref_rslc.getOrbit()

        if self.external_sec_orbit_path is not None:
            self.sec_orbit = load_orbit_from_xml(self.external_sec_orbit_path,
                                                 self.sec_orbit_epoch)
        else:
            self.sec_orbit = self.sec_rslc.getOrbit()

    def add_root_attrs(self):
        """
        Write attributes to the HDF5 root that are common to all InSAR products
        """
        self.attrs["Conventions"] = np.string_("CF-1.7")
        self.attrs["contact"] = np.string_("nisar-sds-ops@jpl.nasa.gov")
        self.attrs["institution"] = np.string_("NASA JPL")
        self.attrs["mission_name"] = np.string_("NISAR")

    def save_to_hdf5(self):
        """
        Write the attributes and groups to the HDF5 file
        """
        self.add_root_attrs()
        self.add_identification_to_hdf5()
        self.add_common_metadata_to_hdf5()
        self.add_procinfo_to_metadata_group()

    def add_procinfo_to_metadata_group(self):
        """
        Add processing information group to metadata group
        """
        self.require_group(self.group_paths.ProcessingInformationPath)
        self.add_algorithms_to_procinfo_group()
        self.add_inputs_to_procinfo_group()
        self.add_parameters_to_procinfo_group()

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithm group to the processing information group
        """
        algo_group = self.require_group(self.group_paths.AlgorithmsPath)

        software_version = DatasetParams(
            "softwareVersion",
            ISCE3_VERSION,
            "Software version used for processing",
        )

        add_dataset_and_attrs(algo_group, software_version)

    def add_baseline_info_to_cubes(self,
                                   cube_group: h5py.Group,
                                   grid: Union[RadarGridParameters,GeoGridParameters],
                                   is_geogrid: bool):
        """
        Add the perpendicular and parallel baseline dataset to either the radarGrid or
        geolocationGrid cubes groups

        Parameters
        ----------
        cube_group : h5py.Group,
            The h5py group object
        grid : object
            radarGridParameters or geogrid object
        is_geogrid : bool
            Flag to indicate if the grid object is geogrid
        """

        # Pull the heights from the radar_grid_cubes group
        # in the runconfig
        radar_grid_cfg = self.cfg["processing"]["radar_grid_cubes"]
        heights = np.array(radar_grid_cfg["heights"])

        # Fetch the baseline information
        baseline_cfg = self.cfg["processing"]['baseline']
        baseline_mode = baseline_cfg['mode']

        # The shape of the baseline
        baseline_ds_shape = [len(heights),
                             grid.length,
                             grid.width]
        if baseline_mode == 'top_bottom':
            baseline_ds_shape[0] = 2

        # Add the baseline dataset to the cube
        for baseline_name in ['parallel', 'perpendicular']:
            ds = \
                cube_group.require_dataset(
                    name= f"{baseline_name}Baseline",
                    shape=baseline_ds_shape,
                    dtype=np.float32)
            ds.attrs['description'] = np.string_(f"{baseline_name.capitalize()}"
                                                 " component of the InSAR baseline")
            ds.attrs['units'] = Units.meter

            # The radarGrid group to attach the x, y, and z coordinates
            if is_geogrid:
                ds.attrs['grid_mapping'] = np.string_('projection')
                ds.dims[1].attach_scale(cube_group['yCoordinates'])
                ds.dims[2].attach_scale(cube_group['xCoordinates'])
                if baseline_mode == '3D_full':
                    ds.dims[0].attach_scale(cube_group['heightAboveEllipsoid'])

    def add_common_to_procinfo_params_group(self):
        """
        Add the common group to the "processingInformation/parameters" group
        """
        for freq, *_ in get_cfg_freq_pols(self.cfg):
            doppler_centroid_path = \
                f"{self.ref_rslc.ProcessingInformationPath}/parameters/frequency{freq}"
            doppler_bandwidth_path = \
                f"{self.ref_rslc.SwathPath}/frequency{freq}"

            doppler_centroid_group = \
                self.ref_h5py_file_obj[doppler_centroid_path]

            doppler_bandwidth_group = \
                self.ref_h5py_file_obj[doppler_bandwidth_path]

            common_group_name = \
                f"{self.group_paths.ParametersPath}/common/frequency{freq}"

            common_group = self.require_group(common_group_name)

            # TODO: the dopplerCentroid and dopplerBandwidth are placeholders heres,
            # and copied from the bandpassed RSLC data.
            # Should those also be updated in the crossmul module?
            doppler_centroid_group.copy("dopplerCentroid", common_group)
            common_group["dopplerCentroid"].attrs['description'] = \
                np.string_("Common Doppler Centroid used for processing interferogram")
            common_group["dopplerCentroid"].attrs['units'] = \
                Units.hertz

            doppler_bandwidth_group.copy(
                "processedAzimuthBandwidth",
                common_group,
                "dopplerBandwidth",
            )
            common_group["dopplerBandwidth"].attrs['description'] = \
                np.string_("Common Doppler Bandwidth used for processing interferogram")
            common_group["dopplerBandwidth"].attrs['units'] = Units.hertz

    def add_RSLC_to_procinfo_params_group(self, rslc_name: str):
        """
        Add the RSLC group to "processingInformation/parameters" group

        Parameters
        ----------
        rslc_name : str
            RSLC name, ('reference' or 'secondary')
        """
        if rslc_name.lower() == "reference":
            rslc_h5py_file_obj = self.ref_h5py_file_obj
            rslc = self.ref_rslc
        else:
            rslc_h5py_file_obj = self.sec_h5py_file_obj
            rslc = self.sec_rslc

        rfi_mit_path = (
            f"{rslc.ProcessingInformationPath}/algorithms/rfiMitigation"
        )
        if rfi_mit_path in rslc_h5py_file_obj:
            rfi_mitigation = rslc_h5py_file_obj[rfi_mit_path][()]
        else:
            rfi_mitigation = None

        rfi_mitigation_flag = str(False)
        if ((rfi_mitigation is not None) and
            (rfi_mitigation.lower() not in ['', 'none', 'disabled'])):
            rfi_mitigation_flag = str(True)

        # get the mixed model and update the description
        mixed_mode = self._get_mixed_mode()
        mixed_mode.description = (f'"True" if {rslc_name} RSLC is a'
                                  ' composite of data collected in multiple'
                                  ' radar modes, "False" otherwise')
        ds_params = [
            DatasetParams(
                "rfiCorrectionApplied",
                rfi_mitigation_flag,
                (
                    "Flag to indicate if RFI correction has been applied"
                    f" to {rslc_name} RSLC"
                ),
            ),
            mixed_mode,
        ]

        dst_param_group = \
            self.require_group(f"{self.group_paths.ParametersPath}/{rslc_name}")

        src_param_group = \
            rslc_h5py_file_obj[f"{rslc.ProcessingInformationPath}/parameters"]

        reference_terrain_height = "referenceTerrainHeight"
        reference_terrain_height_description = \
            f"Reference Terrain Height as a function of time for {rslc_name} RSLC"
        if reference_terrain_height in src_param_group:
            src_param_group.copy(reference_terrain_height, dst_param_group)
            dst_param_group[reference_terrain_height].attrs['description'] = \
                np.string_(reference_terrain_height_description)
            dst_param_group[reference_terrain_height].attrs['units'] = \
                np.string_(Units.meter)
        else:
            ds_param = DatasetParams(
                "referenceTerrainHeight",
                "None",
                reference_terrain_height_description,
                {
                    "units": Units.meter
                },
            )
            add_dataset_and_attrs(dst_param_group, ds_param)

        for ds_param in ds_params:
            add_dataset_and_attrs(dst_param_group, ds_param)

        for freq, *_ in get_cfg_freq_pols(self.cfg):

            rslc_group_frequency_name = \
                f"{self.group_paths.ParametersPath}/{rslc_name}/frequency{freq}"
            rslc_frequency_group = self.require_group(rslc_group_frequency_name)

            swath_frequency_path = f"{rslc.SwathPath}/frequency{freq}/"
            swath_frequency_group = rslc_h5py_file_obj[swath_frequency_path]

            swath_frequency_group.copy("slantRangeSpacing",
                                       rslc_frequency_group)
            rslc_frequency_group['slantRangeSpacing'].attrs['description'] = \
                 f"Slant range spacing of {rslc_name} RSLC"

            # TODO: the rangeBandwidth and azimuthBandwidth are placeholders heres,
            # and copied from the bandpassed RSLC data.
            # Should we update those fields?
            swath_frequency_group.copy(
                "processedRangeBandwidth",
                rslc_frequency_group,
                "rangeBandwidth",
            )
            rslc_frequency_group['rangeBandwidth'].attrs['description'] = \
                 np.string_(f"Processed slant range bandwidth for {rslc_name} RSLC")
            rslc_frequency_group['rangeBandwidth'].attrs['units'] = Units.hertz

            swath_frequency_group.copy(
                "processedAzimuthBandwidth",
                rslc_frequency_group,
                "azimuthBandwidth",
            )
            rslc_frequency_group['azimuthBandwidth'].attrs['description'] = \
                np.string_(f"Processed azimuth bandwidth for {rslc_name} RSLC")
            rslc_frequency_group['azimuthBandwidth'].attrs['units'] = Units.hertz

            swath_group = rslc_h5py_file_obj[rslc.SwathPath]
            swath_group.copy("zeroDopplerTimeSpacing", rslc_frequency_group)
            rslc_frequency_group['zeroDopplerTimeSpacing'].attrs['description'] = \
               np.string_(
                   f"Time interval in the along-track direction for {rslc_name} RSLC raster layers"
               )

            doppler_centroid_group = rslc_h5py_file_obj[
                f"{rslc.ProcessingInformationPath}/parameters/frequency{freq}"
            ]
            doppler_centroid_group.copy(
                "dopplerCentroid", rslc_frequency_group
            )
            rslc_frequency_group['dopplerCentroid'].attrs['description'] = \
                np.string_(f"2D LUT of Doppler centroid for frequency {freq}")
            rslc_frequency_group['dopplerCentroid'].attrs['units'] = Units.hertz

    def add_coregistration_to_algo_group(self):
        """
        Add the coregistration parameters to the
        "processingInfromation/algorithms" group
        """
        proc_cfg = self.cfg["processing"]
        dense_offsets = proc_cfg["dense_offsets"]["enabled"]
        offset_product = proc_cfg["offsets_product"]["enabled"]

        if dense_offsets:
            name = "dense_offsets"
        elif offset_product:
            name = "offsets_product"

        coreg_method = \
            "Coarse geometry coregistration with DEM and orbit ephemeris"
        cross_correlation_domain = "None"
        outlier_filling_method = "None"
        filter_kernel_algorithm = "None"
        culling_metric = "None"

        if dense_offsets or offset_product:
            coreg_method = f"{coreg_method} with cross-correlation refinement"
            cross_correlation_domain = proc_cfg[name][
                "cross_correlation_domain"
            ]
            if proc_cfg["rubbersheet"]["enabled"]:
                outlier_filling_method = \
                    proc_cfg["rubbersheet"]["outlier_filling_method"]

                filter_kernel_algorithm = \
                    proc_cfg["rubbersheet"]["offsets_filter"]

                culling_metric = proc_cfg["rubbersheet"]["culling_metric"]

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
                coreg_method,
                "RSLC coregistration method",
                {
                    "algorithm_type": "RSLC coregistration",
                },
            ),
            DatasetParams(
                "crossCorrelation",
                cross_correlation_domain,
                (
                    "Cross-correlation algorithm for"
                    " sub-pixel offsets computation"
                ),
                {
                    "algorithm_type": "RSLC coregistration",
                },
            ),
            DatasetParams(
                "crossCorrelationFilling",
                outlier_filling_method,
                "Outliers data filling algorithm for"
                " cross-correlation offsets",
                {
                    "algorithm_type": "RSLC coregistration",
                },
            ),
            DatasetParams(
                "crossCorrelationFilterKernel",
                filter_kernel_algorithm,
                "Filtering algorithm for cross-correlation offsets",
                {
                    "algorithm_type": "RSLC coregistration",
                },
            ),
            DatasetParams(
                "crossCorrelationOutliers",
                culling_metric,
                "Outliers identification algorithm",
                {
                    "algorithm_type": "RSLC coregistration",
                },
            ),
            DatasetParams(
                "geometryCoregistration",
                "Range doppler to geogrid then geogrid to range doppler",
                "Geometry coregistration algorithm",
                {
                    "algorithm_type": "RSLC coregistration",
                },
            ),
            DatasetParams(
                "resampling",
                "sinc",
                "Secondary RSLC resampling algorithm",
                {
                    "algorithm_type": "RSLC coregistration",
                },
            ),
        ]

        coregistration_group = self.require_group(
            f"{self.group_paths.AlgorithmsPath}/coregistration")
        for ds_param in algo_coregistration_ds_params:
            add_dataset_and_attrs(coregistration_group, ds_param)


    def add_interferogramformation_to_algo_group(self):
        """
        Add the InterferogramFormation group to "processingInformation/algorithms" group
        """
        flatten_method = "None"
        proc_cfg = self.cfg["processing"]

        if proc_cfg["crossmul"]["flatten"]:
            flatten_method = "With geometry offsets"

        multilooking_method = "Spatial moving average with decimation"
        wrapped_interferogram_filtering_mdethod = \
            proc_cfg["filter_interferogram"]["filter_type"]

        algo_intefergramformation_ds_params = [
            DatasetParams(
                "flatteningMethod",
                flatten_method,
                "Algorithm used to flatten the wrapped interferogram",
                {
                    "algorithm_type": "Interferogram formation",
                },
            ),
            DatasetParams(
                "multilooking",
                multilooking_method,
                "Multilooking algorithm",
                {
                    "algorithm_type": "Interferogram formation",
                },
            ),
            DatasetParams(
                "wrappedInterferogramFiltering",
                wrapped_interferogram_filtering_mdethod,
                (
                    "Algorithm used to filter the wrapped interferogram prior to phase unwrapping"
                ),
                {
                    "algorithm_type": "Interferogram formation",
                },
            ),
        ]

        igram_formation_group = self.require_group(
            f"{self.group_paths.AlgorithmsPath}/"
            "interferogramFormation"
        )
        for ds_param in algo_intefergramformation_ds_params:
            add_dataset_and_attrs(igram_formation_group, ds_param)

    def add_pixeloffsets_to_procinfo_params_group(self):
        """
        Add the pixelOffsets group to "processingInformation/parameters" group
        """
        proc_cfg = self.cfg["processing"]
        is_roff = proc_cfg["offsets_product"]["enabled"]
        merge_gross_offset = get_off_params(
            proc_cfg, "merge_gross_offset", is_roff
        )
        skip_range = get_off_params(proc_cfg, "skip_range", is_roff)
        skip_azimuth = get_off_params(proc_cfg, "skip_azimuth", is_roff)

        half_search_range = get_off_params(
            proc_cfg,
            "half_search_range",
            is_roff,
            pattern="layer",
            get_min=True,
        )
        half_search_azimuth = get_off_params(
            proc_cfg,
            "half_search_azimuth",
            is_roff,
            pattern="layer",
            get_min=True,
        )

        window_azimuth = get_off_params(
            proc_cfg, "window_azimuth", is_roff, pattern="layer", get_min=True
        )
        window_range = get_off_params(
            proc_cfg, "window_range", is_roff, pattern="layer", get_min=True
        )

        oversampling_factor = get_off_params(
            proc_cfg, "correlation_surface_oversampling_factor", is_roff
        )

        pixeloffsets_ds_params = [
            DatasetParams(
                "alongTrackWindowSize",
                np.uint32(window_azimuth),
                "Along-track cross-correlation window size in pixels",
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "slantRangeWindowSize",
                np.uint32(window_range),
                "Slant range cross-correlation window size in pixels",
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "alongTrackSearchWindowSize",
                np.uint32(2 * half_search_azimuth),
                "Along-track cross-correlation search window size in pixels",
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "slantRangeSearchWindowSize",
                np.uint32(2 * half_search_range),
                "Slant range cross-correlation search window size in pixels",
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "alongTrackSkipWindowSize",
                np.uint32(skip_azimuth),
                "Along-track cross-correlation skip window size in pixels",
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "slantRangeSkipWindowSize",
                np.uint32(skip_range),
                "Slant range cross-correlation skip window size in pixels",
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "correlationSurfaceOversampling",
                np.uint32(oversampling_factor),
                "Oversampling factor of the cross-correlation surface",
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "isOffsetsBlendingApplied",
                np.string_(str(merge_gross_offset)),
                (
                    "Flag to indicate if pixel offsets are the results of"
                    " blending multi-resolution layers of pixel offsets"
                ),
            ),
        ]
        for freq, *_ in get_cfg_freq_pols(self.cfg):
            pixeloffsets_group_name = \
                f"{self.group_paths.ParametersPath}/pixelOffsets/frequency{freq}"
            pixeloffsets_group = self.require_group(pixeloffsets_group_name)

            for ds_param in pixeloffsets_ds_params:
                add_dataset_and_attrs(pixeloffsets_group, ds_param)

    def add_parameters_to_procinfo_group(self):
        """
        Add the parameters group to the "processingInformation" group
        """
        params_group = self.require_group(self.group_paths.ParametersPath)

        self.add_common_to_procinfo_params_group()
        self.add_RSLC_to_procinfo_params_group("reference")
        self.add_RSLC_to_procinfo_params_group("secondary")

        runconfig_contents = DatasetParams(
            "runConfigurationContents",
            np.string_(self.cfg),
            (
                "Contents of the run configuration file with parameters"
                " used for processing"
            ),
        )
        add_dataset_and_attrs(params_group, runconfig_contents)

    def add_inputs_to_procinfo_group(self):
        """
        Add the inputs group to the "processingInformation" group
        """
        orbit_file = []
        ancillary_group = self.cfg["dynamic_ancillary_file_group"]
        for idx in ["reference", "secondary"]:
            _orbit_file = \
            ancillary_group["orbit_files"].get(f"{idx}_orbit_file")
            if _orbit_file is None:
                _orbit_file = f"used RSLC internal {idx} orbit file"
            orbit_file.append(_orbit_file)

        # DEM source
        dem_source = \
            ancillary_group["dem_file_description"]

        # if dem source is None, then replace it with None
        if dem_source is None:
            dem_source = "None"

        inputs_ds_params = [
            DatasetParams(
                "configFiles",
                os.path.basename(self.runconfig_path),
                "List of input config files used",
            ),
            DatasetParams(
                "demSource",
                dem_source,
                "Description of the input digital elevation model (DEM)",
            ),
            DatasetParams(
                "l1ReferenceSlcGranules",
                np.string_([os.path.basename(self.ref_h5_slc_file)]),
                "List of input reference L1 RSLC products used",
            ),
            DatasetParams(
                "l1SecondarySlcGranules",
                np.string_([os.path.basename(self.sec_h5_slc_file)]),
                "List of input secondary L1 RSLC products used",
            ),
            DatasetParams(
                "orbitFiles",
                np.string_([orbit_file]),
                "List of input orbit files used",
            ),
        ]
        inputs_group = self.require_group(
            f"{self.group_paths.ProcessingInformationPath}/inputs"
        )
        for ds_param in inputs_ds_params:
            add_dataset_and_attrs(inputs_group, ds_param)

    def add_common_metadata_to_hdf5(self):
        """
        Write metadata datasets and attributes common to all InSAR products to HDF5
        """
        # Can copy entirety of attitude
        ref_metadata_group = self.ref_h5py_file_obj[self.ref_rslc.MetadataPath]
        dst_metadata_group = self.require_group(self.group_paths.MetadataPath)
        ref_metadata_group.copy("attitude", dst_metadata_group)

        # Attitude time
        attitude_time = dst_metadata_group["attitude"]["time"]
        attitude_time_units = attitude_time.attrs['units']
        attitude_time_units = extract_datetime_from_string(str(attitude_time_units),
                                                           'seconds since ')
        if attitude_time_units is not None:
            attitude_time.attrs['units'] = np.string_(attitude_time_units)

        dst_metadata_group["attitude"]["quaternions"].attrs["units"] = \
            Units.unitless
        dst_metadata_group["attitude"]["quaternions"].attrs["eulerAngles"] = \
            Units.radian
        dst_metadata_group["attitude"]["quaternions"].attrs["angularVelocity"] = \
            np.string_("radians / second")

        # Orbit population based in inputs
        if self.external_ref_orbit_path is None:
            ref_metadata_group.copy("orbit", dst_metadata_group)
        else:
            # populate orbit group with contents of external orbit file
            orbit_group = dst_metadata_group.require_group("orbit")
            self.ref_orbit.save_to_h5(orbit_group)

        # Orbit time
        orbit_time = dst_metadata_group["orbit"]["time"]
        orbit_time.attrs['description'] = \
            np.string_("Time vector record. This record contains"
                       " the time corresponding to position and"
                       " velocity records"
                       )

        orbit_time_units = orbit_time.attrs['units']
        orbit_time_units = extract_datetime_from_string(str(orbit_time_units),
                                                        'seconds since ')
        if orbit_time_units is not None:
            orbit_time.attrs['units'] = np.string_(orbit_time_units)
        # Orbit velocity
        dst_metadata_group["orbit"]["velocity"].attrs["units"] = \
            np.string_("meters / second")

    def add_identification_to_hdf5(self):
        """
        Add the identification group to the product
        """
        radar_band_name = self._get_band_name()
        primary_exec_cfg = self.cfg["primary_executable"]

        processing_center = primary_exec_cfg.get("processing_center")
        processing_type = primary_exec_cfg.get("processing_type")
        partial_granule_id = primary_exec_cfg.get("partial_granule_id")
        product_version = primary_exec_cfg.get("product_version")
        crid = primary_exec_cfg.get("composite_release_id")

        # Determine processingType
        if processing_type == 'PR':
            processing_type = np.string_('Nominal')
        elif processing_type == 'UR':
            processing_type = np.string_('Urgent')
        else:
            processing_type = np.string_('Undefined')

        # processing center (JPL, NRSC, or Others)
        # if it is None, 'JPL' will be applied
        if processing_center is None:
            processing_center = "JPL"

        # If the CRID identifier is None, assign a dummy crid "A10000"
        if crid is None:
            crid = "A10000"

        # Extract relevant identification from reference and secondary RSLC
        ref_id_group = self.ref_h5py_file_obj[self.ref_rslc.IdentificationPath]
        sec_id_group = self.sec_h5py_file_obj[self.sec_rslc.IdentificationPath]

        dst_id_group = self.require_group(self.group_paths.IdentificationPath)

        # Datasets that need to be copied from the RSLC
        id_ds_names_need_to_copy = [
            DatasetParams(
                "absoluteOrbitNumber",
                "None",
                "Absolute orbit number",
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "boundingPolygon",
                "None",
                (
                    "OGR compatible WKT representation of bounding polygon of"
                    " the image"
                ),
                {
                    "ogr_geometry": "polygon",
                    "epsg": "4326",
                },
            ),
            DatasetParams(
                "diagnosticModeFlag",
                "None",
                (
                    "Indicates if the radar operation mode is a diagnostic"
                    " mode (1-2) or DBFed science (0): 0, 1, or 2"
                ),
            ),
            DatasetParams(
                "frameNumber",
                "None",
                "Frame number",
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "trackNumber",
                "None",
                "Track number",
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "isDithered",
                "None",
                (
                    '"True" if the pulse timing was varied (dithered) during'
                    ' acquisition, "False" otherwise'
                ),
            ),
            DatasetParams(
                "lookDirection",
                "None",
                'Look direction, either "Left" or "Right"',
            ),
            DatasetParams(
                "missionId",
                "None",
                "Mission identifier",
            ),
            DatasetParams(
                "orbitPassDirection",
                "None",
                'Orbit direction, either "Ascending" or "Descending"',
            ),
            DatasetParams(
                "plannedDatatakeId",
                "None",
                "List of planned datatakes included in the product",
            ),
            DatasetParams(
                "plannedObservationId",
                "None",
                "List of planned observations included in the product",
            ),
            DatasetParams(
                "isUrgentObservation",
                "None",
                'Flag indicating if observation is nominal ("False") or urgent ("True")',
            ),
        ]

        for ds_name in id_ds_names_need_to_copy:
            if ds_name.name in ref_id_group:
                slc_val = ref_id_group[ds_name.name][()]
                ds_name.value = slc_val
            add_dataset_and_attrs(dst_id_group, ds_name)

        # Copy the zero-Dopper information from both reference and secondary RSLC
        for ds_name in ["zeroDopplerStartTime", "zeroDopplerEndTime"]:
            ref_id_group.copy(ds_name, dst_id_group,
                              f"referenceZ{ds_name[1:]}")

            sec_id_group.copy(ds_name, dst_id_group,
                              f"secondaryZ{ds_name[1:]}")

        # Update the description attributes of the zeroDoppler
        for prod in list(product(['reference','secondary'],
                                 ['Start', 'End'])):
            rslc_name, start_or_stop = prod
            ds = dst_id_group[f"{rslc_name}ZeroDoppler{start_or_stop}Time"]
            # rename the End time to stop
            time_in_description = 'stop' if start_or_stop == 'End' else 'start'
            ds.attrs['description'] = \
                f"Azimuth {time_in_description} time of {rslc_name} RSLC product"

        # Granule ID follows the NISAR filename convention. The partial granule ID
        # has placeholders (curly brackets) which will be filled by the InSAR SAS
        # (Partial Granule ID Example:
        # NISAR_{Level}_PR_{ProductType}_001_001_A_001_003_{MODE}_{PO}_A_{StartDateTime}_{EndDateTime}_D00341_P_C_J_001_.h5)

        if partial_granule_id is None:
            granule_id = "None"
        else:
            # Get the first frequency to process and corresponding polarizations
            frequency = list(self.freq_pols.keys())[0]
            granule_id = get_insar_granule_id(self.ref_h5_slc_file, self.sec_h5_slc_file,
                                              partial_granule_id, freq=frequency,
                                              pol_process=self.freq_pols.get(frequency),
                                              product_type=self.product_info.ProductType)
        if product_version is None:
            product_version = \
                self.product_info.ProductVersion

        id_ds_names_to_be_created = [
            DatasetParams(
                "compositeReleaseId",
                np.string_(crid),
                "Unique version identifier of the science data production system",
            ),
            DatasetParams(
                "granuleId",
                granule_id,
                "Unique granule identification name",
            ),
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
                "listOfFrequencies",
                np.string_(list(self.freq_pols)),
                "List of frequency layers available in the product",
            ),
            DatasetParams(
                "processingCenter", processing_center, "Data processing center"
            ),
            DatasetParams(
                "processingDateTime",
                datetime.utcnow().replace(microsecond=0).isoformat(),
                (
                    "Processing UTC date and time in the format YYYY-mm-ddTHH:MM:SS"
                ),
            ),
            DatasetParams(
                "processingType",
                processing_type,
                "Nominal (or) Urgent (or) Custom (or) Undefined",
            ),
            DatasetParams(
                "radarBand", radar_band_name,
                'Acquired frequency band, either "L" or "S"'
            ),
            DatasetParams(
                "productLevel",
                self.product_info.ProductLevel,
                (
                    "Product level. L0A: Unprocessed instrument data; L0B:"
                    " Reformatted, unprocessed instrument data; L1: Processed"
                    " instrument data in radar coordinates system; and L2:"
                    " Processed instrument data in geocoded coordinates system"
                ),
            ),
            DatasetParams(
                "productVersion",
                product_version,
                (
                    "Product version which represents the structure of the"
                    " product and the science content governed by the"
                    " algorithm, input data, and processing parameters"
                ),
            ),
            DatasetParams(
                "productType", self.product_info.ProductType, "Product type"
            ),
            DatasetParams(
                "productSpecificationVersion",
                self.product_info.ProductSpecificationVersion,
                (
                    "Product specification version which represents the schema"
                    " of this product"
                ),
            ),
            DatasetParams(
                "isGeocoded",
                np.string_(str(self.product_info.isGeocoded)),
                'Flag to indicate if the product data is in the radar geometry ("False") '
                'or in the map geometry ("True")',
            ),
        ]
        for ds_param in id_ds_names_to_be_created:
            add_dataset_and_attrs(dst_id_group, ds_param)

    def _pull_pixel_offsets_params(self):
        """
        Pull the pixel offsets parameters from the runconfig dictionary

        Returns
        ----------
        is_roff : boolean
            Offset product or not
        margin : int
            Margin
        rg_start : int
            Start range
        az_start : int
            Start azimuth
        rg_skip : int
            Pixels skiped across range
        az_skip : int
            Pixels skiped across the azimth
        rg_search : int
            Window size across range
        az_search : int
            Window size across azimuth
        rg_chip : int
            Fine window size across range
        az_chip : int
            Fine window size across azimuth
        ovs_factor : int
            Oversampling factor
        """
        proc_cfg = self.cfg["processing"]

        # pull the offset parameters
        is_roff = proc_cfg["offsets_product"]["enabled"]
        (margin, rg_gross, az_gross,
         rg_start, az_start,
         rg_skip, az_skip, ovs_factor) = \
             [get_off_params(proc_cfg, param, is_roff)
              for param in ["margin", "gross_offset_range",
                            "gross_offset_azimuth",
                            "start_pixel_range","start_pixel_azimuth",
                            "skip_range", "skip_azimuth",
                            "correlation_surface_oversampling_factor"]]

        rg_search, az_search, rg_chip, az_chip = \
            [get_off_params(proc_cfg, param, is_roff,
                            pattern="layer",
                            get_min=True,) for param in \
                                ["half_search_range",
                                 "half_search_azimuth",
                                 "window_range",
                                 "window_azimuth"]]
        # Adjust margin
        margin = max(margin, np.abs(rg_gross), np.abs(az_gross))

        # Compute slant range/azimuth vectors of offset grids
        if rg_start is None:
            rg_start = margin + rg_search
        if az_start is None:
            az_start = margin + az_search

        return (is_roff,  margin, rg_start, az_start,
                rg_skip, az_skip, rg_search, az_search,
                rg_chip, az_chip, ovs_factor)

    def _get_band_name(self):
        """
        Get the band name ('L', 'S'), Raises exception if neither is found.

        Returns
        ----------
        str
            'L', 'S'
        """
        freq = "A" if "A" in self.freq_pols else "B"
        swath_frequency_path = f"{self.ref_rslc.SwathPath}/frequency{freq}/"
        freq_group = self.ref_h5py_file_obj[swath_frequency_path]

        # Center frequency in GHz
        center_frequency = freq_group["processedCenterFrequency"][()] / 1e9

        # L band if the center frequency is between 1GHz and 2 GHz
        # S band if the center frequency is between 2GHz and 4 GHz
        # both bands are defined by the IEEE with the reference:
        # https://en.wikipedia.org/wiki/L_band
        # https://en.wikipedia.org/wiki/S_band
        if (center_frequency >= 1.0) and (center_frequency <= 2.0):
            return "L"
        elif (center_frequency > 2.0) and (center_frequency <= 4.0):
            return "S"
        else:
            raise ValueError("Unknown frequency encountered. Not L or S band")

    def _get_mixed_mode(self):
        """
        Determining mixed mode and return result as a DatasetParams

        Returns
        ----------
        isMixedMode : DatasetParams
            DatasetParams object based on bandwidth overlap check
        """
        pols_dict = {}
        for freq, pols, _ in get_cfg_freq_pols(self.cfg):
            pols_dict[freq] = pols

        # Import the check_range_bandwidth_overlap locally to prevent the circular import errors:
        # when import globally, there is the following error.
        # ImportError: cannot import name 'PolChannel' from partially initialized module 'nisar.mixed_mode'
        # (most likely due to a circular import)
        # (/isce/install/packages/nisar/mixed_mode/__init__.py)
        from isce3.splitspectrum.splitspectrum import \
            check_range_bandwidth_overlap

        # Check if there is bandwidth overlap
        mode = check_range_bandwidth_overlap(self.ref_rslc, self.sec_rslc,
                                             pols_dict)

        mixed_mode = False if not mode else True

        return DatasetParams(
            "isMixedMode",
            np.string_(str(mixed_mode)),
            (
                '"True" if this product is generated from reference and'
                ' secondary RSLCs with different range bandwidths, "False"'
                " otherwise"
            ),
        )

    @property
    def default_chunk_size(self):
        """
        Get the default chunk size.
        To change the chunks of the children classes, need to override this property

        Returns
        ----------
        tuple
            (128, 128)
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
        compression_enabled: Optional[bool] = True,
        compression_type: Optional[str] = 'gzip',
        compression_level: Optional[int] = 5,
        chunk_size: Optional[List] = [128, 128],
        shuffle_filter: Optional[bool] = True
    ):
        """
        Create an empty 2D dataset under the h5py.Group

        Parameters
        ----------
        h5_group : h5py.Group
            The parent HDF5 group where the new dataset will be stored
        name : str
            Dataset name
        shape : tuple
            Shape of the dataset
        dtype : object
            Data type of the dataset
        description : str
            Description of the dataset
        units : str, optional
            Units of the dataset
        grid_mapping : str, optional
            Grid mapping string, (e.g. "projection")
        standard_name : str, optional
            Standard name
        long_name : str, optional
            Long name
        yds : h5py.Dataset, optional
            Y coordinates
        xds : h5py.Dataset, optional
            X coordinates
        fill_value : Any, optional
            No data value of the dataset
        compression_enabled: bool
            Flag to enable/disable data compression
        compression_type: str
            Data compression algorithm
        compression_level: int
            Level of data compression (1: low compression, 9: high compression)
        chunk_size: [int, int]
            Chunk size along rows and columns
        shuffle_filter: bool
            Flag to enable/disable shuffle filter
        """
        # Use the minimum size between the predefined chunk size and dataset shape
        # to ensure the dataset is chunked if the compression is enabled.
        ds_chunk_size = tuple(map(min, zip(chunk_size, shape)))

        # Include options for compression
        create_dataset_kwargs = {}
        if compression_enabled:
            if compression_type is not None:
                create_dataset_kwargs['compression'] = compression_type
            if compression_level is not None:
                create_dataset_kwargs['compression_opts'] = compression_level

            # Add shuffle filter options
            create_dataset_kwargs['shuffle'] = shuffle_filter

        if compression_enabled:
            ds = h5_group.require_dataset(
                name, dtype=dtype, shape=shape,
                chunks=ds_chunk_size, **create_dataset_kwargs
            )
        else:
            # create dataset without chunks
            ds = h5_group.require_dataset(name, dtype=dtype, shape=shape,
                                          **create_dataset_kwargs)

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
            ds.attrs["_FillValue"] = fill_value
        # create fill value if not specified
        elif np.issubdtype(dtype, np.floating):
            ds.attrs["_FillValue"] = np.nan
        elif np.issubdtype(dtype, np.unsignedinteger):
            ds.attrs["_FillValue"] = np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.signedinteger):
            ds.attrs["_FillValue"] = np.iinfo(dtype).min
        elif np.issubdtype(dtype, np.complexfloating):
            ds.attrs["_FillValue"] = np.complex64(np.nan + 1j * np.nan)
        elif dtype == complex32:
            ds.attrs["_FillValue"] = \
                to_complex32(np.array([np.nan + 1j * np.nan]))[0]

