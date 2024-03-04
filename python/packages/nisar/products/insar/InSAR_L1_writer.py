import numpy as np
from isce3.core import LUT2d
from isce3.product import RadarGridParameters
from nisar.workflows.h5_prep import (add_geolocation_grid_cubes_to_hdf5,
                                     get_off_params)
from nisar.workflows.helpers import get_cfg_freq_pols

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_base_writer import InSARBaseWriter
from .product_paths import L1GroupsPaths
from .units import Units
from .utils import extract_datetime_from_string, number_to_ordinal


class L1InSARWriter(InSARBaseWriter):
    """
    InSAR Level 1 products writer inherit from the InSARBaseWriter
    The Level 1 products include ROFF, RUNW, and RIFG.

    Attributes
    ----------
    igram_range_looks : int
        range looks for the interferogram
    igram_azimuth_looks : int
        azimuth looks for the interferogram
    """
    def __init__(self, **kwds):
        """
        Constructor for InSAR L1 products (RIFG, RUNW, and ROFF).
        """
        super().__init__(**kwds)

        # Level 1 product group path
        self.group_paths = L1GroupsPaths()

        # Range and azimuth looks that will be performed on the interfergoram
        self.igram_range_looks = 1
        self.igram_azimuth_looks = 1

    def save_to_hdf5(self):
        """
        Write the attributes and groups to the HDF5
        """
        super().save_to_hdf5()

        self.add_geolocation_grid_cubes()
        self.add_swaths_to_hdf5()

    def add_geolocation_grid_cubes(self):
        """
        Add the geolocation grid cubes
        """
        # Pull the heights and espg from the radar_grid_cubes group
        # in the runconfig
        radar_grid_cfg = self.cfg["processing"]["radar_grid_cubes"]
        heights = np.array(radar_grid_cfg["heights"])
        epsg = radar_grid_cfg["output_epsg"]

        # Retrieve the group
        geolocationGrid_path = self.group_paths.GeolocationGridPath
        self.require_group(geolocationGrid_path)

        # Pull the radar frequency
        cube_freq = "A" if "A" in self.freq_pols else "B"
        radargrid = RadarGridParameters(self.ref_h5_slc_file)

        # Figure out decimation factors that give < 500 m spacing.
        # NOTE: same with the RSLC, the max spacing = 500m is hardcoded.
        # In addition, we cannot have the exact 500m spacing for both azimuth
        # and range directions.
        max_spacing = 500.0
        t = radargrid.sensing_mid + \
            (radargrid.ref_epoch - self.orbit.reference_epoch).total_seconds()

        _, v = self.orbit.interpolate(t)
        dx = np.linalg.norm(v) / radargrid.prf

        grid_doppler = LUT2d()
        native_doppler = self.ref_rslc.getDopplerCentroid(
            frequency=cube_freq
        )

        native_doppler.bounds_error = False

        geo2rdr_params = dict(threshold_geo2rdr=1e-8,
                              numiter_geo2rdr=50,
                              delta_range=10)

        # Create a new geolocation radar grid with 5 extra points
        # before and after the starting and ending
        # zeroDopplerTime and slantRange
        extra_points = 5

        # Total number of samples along the azimuth and slant range
        # using around 500m sampling interval
        ysize = int(np.ceil(radargrid.length / (max_spacing / dx)))
        xsize = int(np.ceil(radargrid.width / \
            (max_spacing / radargrid.range_pixel_spacing)))

        # New geolocation grid
        geolocation_radargrid = \
            radargrid.resize_and_keep_startstop(ysize, xsize)
        geolocation_radargrid = \
            geolocation_radargrid.add_margin(extra_points,
                                             extra_points)

        # Add geolocation grid cubes to hdf5
        add_geolocation_grid_cubes_to_hdf5(
            self,
            geolocationGrid_path,
            geolocation_radargrid,
            heights,
            self.orbit,
            native_doppler,
            grid_doppler,
            epsg,
            **geo2rdr_params,
        )

        # Add the min and max attributes to the following dataset
        ds_names = ["incidenceAngle",
                    "losUnitVectorX",
                    "losUnitVectorY",
                    "alongTrackUnitVectorX",
                    "alongTrackUnitVectorY",
                    "elevationAngle"]

        geolocation_grid_group = self[geolocationGrid_path]
        for ds_name in ds_names:
            ds = geolocation_grid_group[ds_name][()]
            valid_min, valid_max = np.nanmin(ds), np.nanmax(ds)
            geolocation_grid_group[ds_name].attrs["min"] = valid_min
            geolocation_grid_group[ds_name].attrs["max"] = valid_max

        geolocation_grid_group['epsg'][...] = \
            geolocation_grid_group['epsg'][()].astype(np.uint32)
        geolocation_grid_group['epsg'].attrs['description'] = \
            np.string_("EPSG code corresponding to the coordinate system"
                       " used for representing the geolocation grid")
        geolocation_grid_group['epsg'].attrs['units'] = Units.unitless
        geolocation_grid_group['losUnitVectorX'].attrs['units'] = Units.unitless
        geolocation_grid_group['losUnitVectorY'].attrs['units'] = Units.unitless
        geolocation_grid_group['losUnitVectorY'].attrs['description'] = \
            np.string_("LOS unit vector Y")

        geolocation_grid_group['alongTrackUnitVectorX'].attrs['units'] = \
            Units.unitless
        geolocation_grid_group['alongTrackUnitVectorY'].attrs['units'] = \
            Units.unitless
        geolocation_grid_group['heightAboveEllipsoid'][...] = \
            geolocation_grid_group['heightAboveEllipsoid'][()].astype(np.float64)

        zero_dopp_time_units = geolocation_grid_group['zeroDopplerTime'].attrs['units']
        zero_dopp_time_units = extract_datetime_from_string(str(zero_dopp_time_units),
                                                            'seconds since ')
        if zero_dopp_time_units is not None:
            geolocation_grid_group['zeroDopplerTime'].attrs['units']\
                = np.string_(zero_dopp_time_units)

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithms group to the processingInformation group
        """
        super().add_algorithms_to_procinfo_group()
        self.add_coregistration_to_algo_group()

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
                np.string_(str(range_filter)),
                (
                    "Flag to indicate if common band range filter has been"
                    " applied"
                ),
            ),
            DatasetParams(
                "commonBandAzimuthFilterApplied",
                np.string_(str(azimuth_filter)),
                (
                    "Flag to indicate if common band azimuth filter has been"
                    " applied"
                ),
            ),
            DatasetParams(
                "ellipsoidalFlatteningApplied",
                np.string_(str(flatten)),
                (
                    "Flag to indicate if the interferometric phase has been "
                    "flattened with respect to a zero height ellipsoid"
                ),
            ),
            DatasetParams(
                "topographicFlatteningApplied",
                np.string_(str(flatten)),
                (
                    "Flag to indicate if the interferometric phase has been "
                    "flattened with respect to topographic height using a DEM"
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
                    "units": Units.unitless,
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
                    "units": Units.unitless,
                },
            ),
        ]

        for freq, *_ in get_cfg_freq_pols(self.cfg):
            bandwidth_group_path = f"{self.ref_rslc.SwathPath}/frequency{freq}"
            bandwidth_group = self.ref_h5py_file_obj[bandwidth_group_path]

            igram_group_name = \
                f"{self.group_paths.ParametersPath}/interferogram/frequency{freq}"
            igram_group = self.require_group(igram_group_name)

            # TODO: the azimuthBandwidth and rangeBandwidth are placeholders heres,
            # and copied from the bandpassed RSLC data.
            # those should be updated in the crossmul module.
            bandwidth_group.copy(
                "processedAzimuthBandwidth",
                igram_group,
                "azimuthBandwidth",
            )
            igram_group['azimuthBandwidth'].attrs['description'] = \
                np.string_("Processed azimuth bandwidth for frequency " + \
                           f"{freq} interferometric layers")
            igram_group['azimuthBandwidth'].attrs['units'] = Units.hertz

            bandwidth_group.copy(
                "processedRangeBandwidth",
                igram_group,
                "rangeBandwidth",
            )
            igram_group['rangeBandwidth'].attrs['description'] = \
                np.string_("Processed slant range bandwidth for frequency " + \
                           f"{freq} interferometric layers")
            igram_group['rangeBandwidth'].attrs['units'] = Units.hertz

            for ds_param in interferogram_ds_params:
                add_dataset_and_attrs(igram_group, ds_param)

    def add_parameters_to_procinfo_group(self):
        """
        Add the parameters group to the "processingInformation" group
        """
        super().add_parameters_to_procinfo_group()
        self.add_interferogram_to_procinfo_params_group()
        self.add_pixeloffsets_to_procinfo_params_group()

    def _get_interferogram_dataset_shape(self, freq : str, pol : str):
        """
        Get the interfergraom dataset shape at a given frequency and polarization

        Parameters
        ---------
        freq: str
            frequency ('A' or 'B')
        pol : str
            polarization ('HH', 'HV', 'VH', or 'VV')

        Returns
        ----------
        igram_shape : tuple
             interfergraom shape
        """
        # get the RSLC lines and columns
        slc_dset = \
            self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}/{pol}"]
        slc_lines, slc_cols = slc_dset.shape

        # shape of the interferogram product
        igram_shape = (slc_lines // self.igram_azimuth_looks,
                       slc_cols // self.igram_range_looks)

        return igram_shape

    def _get_pixeloffsets_dataset_shape(self, freq : str, pol : str):
        """
        Get the pixel offsets dataset shape at a given frequency and polarization

        Parameters
        ---------
        freq: str
            frequency ('A' or 'B')
        pol : str
            polarization ('HH', 'HV', 'VH', or 'VV')

        Returns
        ----------
        tuple
            (off_length, off_width):
        """
        proc_cfg = self.cfg["processing"]
        is_roff,  margin, _, _,\
        rg_skip, az_skip, rg_search, az_search,\
        rg_chip, az_chip, _ = self._pull_pixel_offsets_params()

        # get the RSLC lines and columns
        slc_dset = \
            self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}/{pol}"]

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
        return (off_length, off_width)


    def _add_datasets_to_pixel_offset_group(self):
        """
        Add datasets to pixel offsets group
        """
        pcfg = self.cfg['processing']
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # create the swath group
            swaths_freq_group_name = \
                f"{self.group_paths.SwathsPath}/frequency{freq}"

            # get the shape of offset product
            off_shape = self._get_pixeloffsets_dataset_shape(freq,
                                                             pol_list[0])

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
                        "Along-track offset",
                        Units.meter,
                    ),
                    (
                        "correlationSurfacePeak",
                        "Normalized correlation surface peak",
                        Units.unitless,
                    ),
                    (
                        "slantRangeOffset",
                        "Slant range offset",
                        Units.meter,
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
                        compression_enabled=self.cfg['output']['compression_enabled'],
                        compression_level=self.cfg['output']['compression_level'],
                        chunk_size=self.cfg['output']['chunk_size'],
                        shuffle_filter=self.cfg['output']['shuffle']
                    )

    def add_pixel_offsets_to_swaths_group(self):
        """
        Add pixel offsets product to swaths group
        """
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

            # shape of offset product
            off_length, off_width = self._get_pixeloffsets_dataset_shape(freq,
                                                                         pol_list[0])

            # add the slantRange, zeroDopplerTime, and their spacings to pixel offset group
            offset_slant_range = \
                rslc_freq_group["slantRange"][()][rg_start::rg_skip][:off_width]
            offset_zero_doppler_time = \
                rslc_swaths_group["zeroDopplerTime"][()][az_start::az_skip][:off_length]
            offset_zero_doppler_time_spacing = \
                rslc_swaths_group["zeroDopplerTimeSpacing"][()] * az_skip
            offset_slant_range_spacing = \
                rslc_freq_group["slantRangeSpacing"][()] * rg_skip

            zero_dopp_time_units = \
                rslc_swaths_group["zeroDopplerTime"].attrs['units']
            time_str = extract_datetime_from_string(str(zero_dopp_time_units),
                                                    'seconds since ')
            if time_str is not None:
                zero_dopp_time_units = time_str

            ds_offsets_params = [
                DatasetParams(
                    "slantRange",
                    offset_slant_range,
                    "Slant range vector",
                    {'units': Units.meter},
                ),
                DatasetParams(
                    "zeroDopplerTime",
                    offset_zero_doppler_time,
                    "Zero Doppler azimuth time vector",
                    {'units': zero_dopp_time_units},
                ),
                DatasetParams(
                    "zeroDopplerTimeSpacing",
                    offset_zero_doppler_time_spacing,
                    "Along-track spacing of the offset grid",
                    {'units': Units.second},
                ),
                DatasetParams(
                    "slantRangeSpacing",
                    offset_slant_range_spacing,
                    "Slant range spacing of the offset grid",
                    {'units': Units.meter},
                ),
            ]
            offset_group_name = f"{swaths_freq_group_name}/pixelOffsets"
            offset_group = self.require_group(offset_group_name)
            for ds_param in ds_offsets_params:
                add_dataset_and_attrs(offset_group, ds_param)

        # add the datasets to pixel offsets group
        self._add_datasets_to_pixel_offset_group()

    def add_interferogram_to_swaths_group(self):
        """
        Add the interferogram group to the swaths group
        """
        pcfg = self.cfg['processing']
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
                    * self.igram_azimuth_looks,
                    (
                        "Nominal along-track spacing in meters "
                        "between consecutive lines near mid-swath of the product images"
                    ),
                    {"units": Units.meter},
                ),
                DatasetParams(
                    "sceneCenterGroundRangeSpacing",
                    rslc_freq_group["sceneCenterGroundRangeSpacing"][()]
                    * self.igram_range_looks,
                    (
                        "Nominal ground range spacing in meters between "
                        "consecutive pixels near mid-swath of the product images"
                    ),
                    {"units": Units.meter},
                ),
            ]
            for ds_param in scene_center_params:
                add_dataset_and_attrs(swaths_freq_group, ds_param)

            # shape of the interferogram product
            igram_shape = self._get_interferogram_dataset_shape(freq,
                                                                pol_list[0])

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
                    (igram_slant_range, self.igram_range_looks),
                    (igram_zero_doppler_time, self.igram_azimuth_looks),
                )
            )

            igram_slant_range = igram_slant_range[rg_idx]
            igram_zero_doppler_time = igram_zero_doppler_time[az_idx]
            igram_zero_doppler_time_spacing = \
                rslc_swaths_group["zeroDopplerTimeSpacing"][()] * \
                    self.igram_azimuth_looks
            igram_slant_range_spacing = \
                rslc_freq_group["slantRangeSpacing"][()] * \
                    self.igram_range_looks

            zero_dopp_time_units = \
                rslc_swaths_group["zeroDopplerTime"].attrs['units']
            time_str = extract_datetime_from_string(str(zero_dopp_time_units),
                                                    'seconds since ')
            if time_str is not None:
                zero_dopp_time_units = time_str

            ds_igram_params = [
                DatasetParams(
                    "slantRange",
                    igram_slant_range,
                    "Slant range vector",
                    {'units': Units.meter},
                ),
                DatasetParams(
                    "zeroDopplerTime",
                    igram_zero_doppler_time,
                    "Zero Doppler azimuth time vector",
                    {'units': zero_dopp_time_units},
                ),
                DatasetParams(
                    "zeroDopplerTimeSpacing",
                    igram_zero_doppler_time_spacing,
                    (
                        "Time interval in the along-track direction for raster"
                        " layers. This is same as the spacing between"
                        " consecutive entries in the zeroDopplerTime array"
                    ),
                    {'units': Units.second},
                ),
                DatasetParams(
                    "slantRangeSpacing",
                    igram_slant_range_spacing,
                    (
                        "Slant range spacing of grid. Same as difference"
                        " between consecutive samples in slantRange array"
                    ),
                    {'units': Units.meter},
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
                        Units.unitless,
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
                        compression_enabled=self.cfg['output']['compression_enabled'],
                        compression_level=self.cfg['output']['compression_level'],
                        chunk_size=self.cfg['output']['chunk_size'],
                        shuffle_filter=self.cfg['output']['shuffle']
                    )


    def add_subswaths_to_swaths_group(self):
        """
        Add subswaths to the swaths group
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
            number_of_subswaths = rslc_freq_group["numberOfSubSwaths"]
            number_of_subwaths_ds = \
                swaths_freq_group.require_dataset("numberOfSubSwaths",
                                                  shape=number_of_subswaths.shape,
                                                  dtype=np.uint8,
                                                  data=number_of_subswaths[...])
            number_of_subwaths_ds.attrs['description'] = \
                np.string_('Number of swaths of continuous imagery, due to transmit gaps')
            number_of_subwaths_ds.attrs['units'] = Units.unitless

            # valid samples subswath
            num_of_subswaths = rslc_freq_group["numberOfSubSwaths"][()]
            for sub in range(num_of_subswaths):
                subswath = sub + 1
                # Get RSLC subswath dataset, range looks, and destination
                # dataset name based on keys in RSLC
                valid_samples_subswath_name = f"validSamplesSubSwath{subswath}"
                description = \
                    "First and last valid sample in each line of" +\
                    f" {number_to_ordinal(subswath)} subswath"

                if valid_samples_subswath_name in rslc_freq_group.keys():
                    rslc_freq_subswath_ds = \
                        rslc_freq_group[valid_samples_subswath_name]
                    number_of_range_looks =rslc_freq_subswath_ds[()] \
                            // self.igram_range_looks
                else:
                    rslc_freq_subswath_ds = rslc_freq_group["validSamples"]
                    number_of_range_looks = rslc_freq_subswath_ds[()] // \
                        self.igram_range_looks
                    valid_samples_subswath_name = "validSamples"

                # Create subswath dataset and update attributes from RSLC
                dst_subswath_ds = swaths_freq_group.require_dataset(
                    name=valid_samples_subswath_name,
                    data=number_of_range_looks,
                    shape=number_of_range_looks.shape,
                    dtype=np.uint32,
                )
                dst_subswath_ds.attrs['units'] = Units.unitless
                dst_subswath_ds.attrs['description'] = np.string_(description)


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
                f"List of processed polarization layers with frequency {freq}",
            )
            add_dataset_and_attrs(swaths_freq_group, list_of_pols)

            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"]
            rslc_freq_group.copy(
                "processedCenterFrequency",
                swaths_freq_group,
                "centerFrequency",
            )

            # Add the description and units
            cfreq = swaths_freq_group["centerFrequency"]
            cfreq.attrs['description'] = np.string_("Center frequency of"
                                                    " the processed image in hertz")
            cfreq.attrs['units'] = Units.hertz

        self.add_pixel_offsets_to_swaths_group()
