import numpy as np
from isce3.core import LUT2d
from isce3.product import RadarGridParameters
from nisar.workflows.h5_prep import (add_geolocation_grid_cubes_to_hdf5,
                                     get_off_params)
from nisar.workflows.helpers import get_cfg_freq_pols

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_base_writer import InSARWriter
from .product_paths import L1GroupsPaths


class L1InSARWriter(InSARWriter):
    """
    InSAR Level 1 prodcuts (e.g. RIFG, RUNW, ROFF) writer inherenting from the InSARWriter
    """

    def __init__(self, **kwds):
        """
        Constructor for InSAR L1 product (RIFG, RUNW, and ROFF).
        """
        super().__init__(**kwds)

        # Level 1 product group path
        self.group_paths = L1GroupsPaths()

    def save_to_hdf5(self):
        """
        write the attributes and groups to the HDF5
        """
        super().save_to_hdf5()

        self.add_geolocation_grid_cubes()
        self.add_swaths_to_hdf5()

    def add_geolocation_grid_cubes(self):
        """
        Add the geolocation grid cubes
        """

        # Retrieve the group
        geolocationGrid_path = self.group_paths.GeolocationGridPath
        self.require_group(geolocationGrid_path)

        # Pull the radar frequency
        cube_freq = "A" if "A" in self.freq_pols else "B"
        radargrid = RadarGridParameters(self.ref_h5_slc_file)

        # Default is [-500, 9000] meters
        heights = np.linspace(-500, 9000, 20)

        # Figure out decimation factors that give < 500 m spacing.
        max_spacing = 500.0
        t = radargrid.sensing_mid + \
            (radargrid.ref_epoch - self.orbit.reference_epoch).total_seconds()

        _, v = self.orbit.interpolate(t)
        dx = np.linalg.norm(v) / radargrid.prf
        tskip = int(np.floor(max_spacing / dx))
        rskip = int(np.floor(max_spacing / radargrid.range_pixel_spacing))
        radargrid = radargrid[::tskip, ::rskip]

        grid_doppler = LUT2d()
        cube_native_doppler = self.ref_rslc.getDopplerCentroid(
            frequency=cube_freq
        )
        cube_native_doppler.bounds_error = False

        tol = dict(
            threshold_geo2rdr=1e-8,
            numiter_geo2rdr=50,
            delta_range=10,
        )

        # Add geolocation grid cubes to hdf5
        add_geolocation_grid_cubes_to_hdf5(
            self,
            geolocationGrid_path,
            radargrid,
            heights,
            self.orbit,
            cube_native_doppler,
            grid_doppler,
            4326,
            **tol,
        )

        # Add the min and max attributes to the following dataset
        ds_names = [
            "incidenceAngle",
            "losUnitVectorX",
            "losUnitVectorY",
            "alongTrackUnitVectorX",
            "alongTrackUnitVectorY",
            "elevationAngle",
        ]
        geolocation_grid_group = self[geolocationGrid_path]
        for ds_name in ds_names:
            ds = geolocation_grid_group[ds_name][()]
            valid_min, valid_max = np.nanmin(ds), np.nanmax(ds)
            geolocation_grid_group[ds_name].attrs["min"] = valid_min
            geolocation_grid_group[ds_name].attrs["max"] = valid_max

    def add_algorithms_to_procinfo(self):
        """
        Add the algorithms group to the processingInformation group

        Returns
        ----------
        algo_group : h5py.Group
            The algorithm group object
        """
        
        algo_group = super().add_algorithms_to_procinfo()
        self.add_coregistration_to_algo(algo_group)
        
        return algo_group

    def add_parameters_to_procinfo(self):
        """
        Add the parameters group to the "processingInformation" group
        """

        super().add_parameters_to_procinfo()

        self.add_interferogram_to_procinfo_params()
        self.add_pixeloffsets_to_procinfo_params()

    def _add_datasets_to_pixel_offset(self):
        """
        Add datasets to pixel offsets group
        """
        
        proc_cfg = self.cfg["processing"]
        is_roff,  margin, rg_start, az_start,\
        rg_skip, az_skip, rg_search, az_search,\
        rg_chip, az_chip, _ = self._pull_pixel_offsets_params()     

        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            swaths_freq_group_name = (
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            )
            
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

            # add the interferogram and pixelOffsets groups to the polarization group
            for pol in pol_list:
  
                offset_pol_group_name = (
                    f"{swaths_freq_group_name}/pixelOffsets/{pol}"
                )
                offset_pol_group = self.require_group(offset_pol_group_name)

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
                            
    def add_pixel_offsets_to_swaths(self):
        """
        Add pixel offsets product to swaths group
        """
        
        proc_cfg = self.cfg["processing"]

        is_roff,  margin, rg_start, az_start,\
        rg_skip, az_skip, rg_search, az_search,\
        rg_chip, az_chip, _ = self._pull_pixel_offsets_params()     

        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            swaths_freq_group_name = \
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            
            swaths_freq_group = self.require_group(swaths_freq_group_name)

            # center frequency and sub swaths groups of the RSLC
            rslc_swaths_group = \
                self.ref_h5py_file_obj[f"{self.ref_rslc.SwathPath}"]

            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"
            ]

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

        # add the datasets to pixel offsets group
        self._add_datasets_to_pixel_offset()
        
    def add_interferogram_to_swaths(self, rg_looks: int, az_looks: int):
        """
        Add the interferogram group to the swaths group
        
        Parameters
        ----------
        rg_looks : int
            range looks
        az_looks : int
            azimuth looks
        """
  
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

            # add scene center parameters
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
                
            # get the RSLC lines and columns
            slc_dset = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}/{pol_list[0]}"
            ]
            slc_lines, slc_cols = slc_dset.shape

            # shape of the interferogram product
            igram_shape = (slc_lines // az_looks, slc_cols // rg_looks)

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

            # add the interferogram and pixelOffsets groups to the polarization group
            for pol in pol_list:
                igram_pol_group_name = (
                    f"{swaths_freq_group_name}/interferogram/{pol}"
                )
                igram_pol_group = self.require_group(igram_pol_group_name)

                # Interferogram datasets
                igram_ds_params = [
                    (
                        "coherenceMagnitude",
                        np.float32,
                        f"Coherence magnitude between {pol} layers",
                        "unitless",
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

    
    def add_subswaths_to_swaths(self, rg_looks: int):
        """
        Add subswaths to the swaths group
        
        Parameters
        ----------
        rg_looks : int
            range looks
        """
        
        for freq, *_ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            swaths_freq_group_name = (
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            )
            swaths_freq_group = self.require_group(swaths_freq_group_name)

            # Sub swaths groups of the RSLC
            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"
            ]
            rslc_freq_group.copy("numberOfSubSwaths", swaths_freq_group)

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
    
    
    def add_swaths_to_hdf5(self):
        """
        Add Swaths to the HDF5
        """
        self.require_group(self.group_paths.SwathsPath)

        # Add the common datasetst to the swaths group
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            swaths_freq_group_name = (
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            )
            swaths_freq_group = self.require_group(swaths_freq_group_name)
            
            list_of_pols = DatasetParams(
                "listOfPolarizations",
                np.string_(pol_list),
                f"List of processed polarization layers with frequency{freq}",
            )
            add_dataset_and_attrs(swaths_freq_group, list_of_pols)
            
            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"]
            rslc_freq_group.copy(
                "processedCenterFrequency",
                swaths_freq_group,
                "centerFrequency",
            )
        
        self.add_pixel_offsets_to_swaths()