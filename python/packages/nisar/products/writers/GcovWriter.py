import os
import journal
import isce3
import h5py
import tempfile
from osgeo import gdal
import numpy as np

import nisar.workflows.helpers as helpers
from nisar.products.writers import BaseL2WriterSingleInput
from nisar.products.writers.BaseL2WriterSingleInput import save_dataset
from nisar.workflows.h5_prep import set_get_geo_info


def _save_list_cov_terms(cov_elements_list, dataset_group):

    name = "listOfCovarianceTerms"
    cov_elements_list.sort()
    cov_elements_array = np.array(cov_elements_list, dtype="S4")
    dset = dataset_group.create_dataset(name, data=cov_elements_array)
    desc = "List of processed covariance terms"
    dset.attrs["description"] = np.bytes_(desc)


def run_geocode_cov(cfg, hdf5_obj, root_ds,
                    frequency, pol_list,
                    radar_grid, input_raster_list,
                    grid_doppler,
                    raster_scratch_dir,
                    geogrid, orbit, gcov_terms_file_extension,
                    output_gcov_terms_raster_files_format,
                    secondary_layers_file_extension,
                    secondary_layer_files_raster_files_format,
                    flag_fullcovariance, radar_grid_nlooks,
                    output_gcov_terms_kwargs,
                    output_secondary_layers_kwargs,
                    optional_geo_kwargs):

    error_channel = journal.error("run_geocode_cov")

    # DEM parameters
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']
    dem_interp_method_enum = cfg['processing']['dem_interpolation_method_enum']

    # unpack RTC run parameters
    rtc_dict = cfg['processing']['rtc']
    output_terrain_radiometry = rtc_dict['output_type_enum']
    rtc_algorithm = rtc_dict['algorithm_type_enum']
    input_terrain_radiometry = rtc_dict['input_terrain_radiometry_enum']
    rtc_min_value_db = rtc_dict['rtc_min_value_db']
    rtc_upsampling = rtc_dict['dem_upsampling']

    rtc_area_beta_mode = rtc_dict['area_beta_mode']
    if rtc_area_beta_mode == 'pixel_area':
        rtc_area_beta_mode_enum = \
            isce3.geometry.RtcAreaBetaMode.PIXEL_AREA
    elif rtc_area_beta_mode == 'projection_angle':
        rtc_area_beta_mode_enum = \
            isce3.geometry.RtcAreaBetaMode.PROJECTION_ANGLE
    elif (rtc_area_beta_mode == 'auto' or
            rtc_area_beta_mode is None):
        rtc_area_beta_mode_enum = \
            isce3.geometry.RtcAreaBetaMode.AUTO
    else:
        err_msg = ('ERROR invalid area beta mode:'
                   f' {rtc_area_beta_mode}')
        raise ValueError(err_msg)

    # unpack geocode run parameters
    geocode_dict = cfg['processing']['geocode']
    geocode_algorithm = geocode_dict['algorithm_type']
    output_mode = geocode_dict['output_mode']
    flag_apply_rtc = geocode_dict['apply_rtc']

    apply_valid_samples_sub_swath_masking = \
        geocode_dict['apply_valid_samples_sub_swath_masking']
    memory_mode = geocode_dict['memory_mode_enum']
    save_nlooks = geocode_dict['save_nlooks']
    save_rtc_anf, save_rtc_anf_gamma0_to_sigma0 = \
        read_and_validate_rtc_anf_flags(geocode_dict, flag_apply_rtc,
                                        output_terrain_radiometry)
    save_mask = geocode_dict['save_mask']
    save_dem = geocode_dict['save_dem']

    min_block_size_mb = cfg["processing"]["geocode"]['min_block_size']
    max_block_size_mb = cfg["processing"]["geocode"]['max_block_size']

    # optional keyword arguments , i.e. arguments that may or may not be
    # included in the call to geocode()
    optional_geo_kwargs = {}

    # read min/max block size converting MB to B
    if min_block_size_mb is not None:
        optional_geo_kwargs['min_block_size'] = min_block_size_mb * (2**20)
    if max_block_size_mb is not None:
        optional_geo_kwargs['max_block_size'] = max_block_size_mb * (2**20)

    # unpack geo2rdr parameters
    geo2rdr_dict = cfg['processing']['geo2rdr']
    threshold = geo2rdr_dict['threshold']
    maxiter = geo2rdr_dict['maxiter']

    if (flag_apply_rtc and output_terrain_radiometry ==
            isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT):
        output_radiometry_str = "radar backscatter sigma0"
    elif (flag_apply_rtc and output_terrain_radiometry ==
            isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT):
        output_radiometry_str = 'radar backscatter gamma0'
    elif input_terrain_radiometry == \
            isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT:
        output_radiometry_str = 'radar backscatter beta0'
    else:
        output_radiometry_str = 'radar backscatter sigma0'

    dem_raster = isce3.io.Raster(dem_file)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # set paths temporary files
    input_temp = tempfile.NamedTemporaryFile(
        dir=raster_scratch_dir, suffix='.vrt')
    input_raster_obj = isce3.io.Raster(
        input_temp.name, raster_list=input_raster_list)

    # init Geocode object depending on raster type
    if input_raster_obj.datatype() == gdal.GDT_Float32:
        geo = isce3.geocode.GeocodeFloat32()
    elif input_raster_obj.datatype() == gdal.GDT_Float64:
        geo = isce3.geocode.GeocodeFloat64()
    elif input_raster_obj.datatype() == gdal.GDT_CFloat32:
        geo = isce3.geocode.GeocodeCFloat32()
    elif input_raster_obj.datatype() == gdal.GDT_CFloat64:
        geo = isce3.geocode.GeocodeCFloat64()
    else:
        err_str = 'Unsupported raster type for geocoding'
        error_channel.log(err_str)
        raise NotImplementedError(err_str)

    # init geocode members
    geo.orbit = orbit
    geo.ellipsoid = ellipsoid
    geo.doppler = grid_doppler
    geo.threshold_geo2rdr = threshold
    geo.numiter_geo2rdr = maxiter

    # set data interpolator based on the geocode algorithm
    if output_mode == isce3.geocode.GeocodeOutputMode.INTERP:
        geo.data_interpolator = geocode_algorithm

    geo.geogrid(geogrid.start_x, geogrid.start_y,
                geogrid.spacing_x, geogrid.spacing_y,
                geogrid.width, geogrid.length, geogrid.epsg)

    # create a NamedTemporaryFile and an ISCE3 Raster object to
    # temporarily hold the output imagery
    temp_output = tempfile.NamedTemporaryFile(
        dir=raster_scratch_dir, suffix=gcov_terms_file_extension)

    output_raster_obj = isce3.io.Raster(
        temp_output.name,
        geogrid.width, geogrid.length,
        input_raster_obj.num_bands,
        gdal.GDT_Float32, output_gcov_terms_raster_files_format)

    # create a NamedTemporaryFile and an ISCE3 Raster object to
    # temporarily hold the off-diagonal terms (if applicable)
    nbands_off_diag_terms = 0
    out_off_diag_terms_obj = None
    if flag_fullcovariance:
        nbands = input_raster_obj.num_bands
        nbands_off_diag_terms = (nbands**2 - nbands) // 2
        if nbands_off_diag_terms > 0:
            temp_off_diag = tempfile.NamedTemporaryFile(
                dir=raster_scratch_dir,
                suffix=gcov_terms_file_extension)
            out_off_diag_terms_obj = isce3.io.Raster(
                temp_off_diag.name,
                geogrid.width, geogrid.length,
                nbands_off_diag_terms,
                gdal.GDT_CFloat32, output_gcov_terms_raster_files_format)

    # create a NamedTemporaryFile and an ISCE3 Raster object to
    # temporarily hold the number of looks layer
    if save_nlooks:
        temp_nlooks = tempfile.NamedTemporaryFile(
            dir=raster_scratch_dir,
            suffix=secondary_layers_file_extension)
        out_geo_nlooks_obj = isce3.io.Raster(
            temp_nlooks.name,
            geogrid.width, geogrid.length, 1,
            gdal.GDT_Float32, secondary_layer_files_raster_files_format)
    else:
        temp_nlooks = None
        out_geo_nlooks_obj = None

    # create a NamedTemporaryFile and an ISCE3 Raster object to
    # temporarily hold the radiometric terrain correction (RTC)
    # area normalization factor (ANF) layer
    if save_rtc_anf:
        temp_rtc_anf = tempfile.NamedTemporaryFile(
            dir=raster_scratch_dir,
            suffix=secondary_layers_file_extension)
        out_geo_rtc_obj = isce3.io.Raster(
            temp_rtc_anf.name,
            geogrid.width, geogrid.length, 1,
            gdal.GDT_Float32, secondary_layer_files_raster_files_format)
    else:
        temp_rtc_anf = None
        out_geo_rtc_obj = None

    # create a NamedTemporaryFile and an ISCE3 Raster object to
    # temporarily hold the layer to convert gamma0 backscatter into
    # sigma0
    if save_rtc_anf_gamma0_to_sigma0:
        temp_rtc_anf_gamma0_to_sigma0 = tempfile.NamedTemporaryFile(
            dir=raster_scratch_dir,
            suffix=secondary_layers_file_extension)
        out_geo_rtc_gamma0_to_sigma0_obj = isce3.io.Raster(
            temp_rtc_anf_gamma0_to_sigma0.name,
            geogrid.width, geogrid.length, 1,
            gdal.GDT_Float32, secondary_layer_files_raster_files_format)
    else:
        temp_rtc_anf_gamma0_to_sigma0 = None
        out_geo_rtc_gamma0_to_sigma0_obj = None

    # create a NamedTemporaryFile and an ISCE3 Raster object to
    # temporarily hold the interpolated DEM layer
    if save_dem:
        temp_interpolated_dem = tempfile.NamedTemporaryFile(
            dir=raster_scratch_dir,
            suffix=secondary_layers_file_extension)
        if (output_mode ==
                isce3.geocode.GeocodeOutputMode.AREA_PROJECTION):
            interpolated_dem_width = geogrid.width + 1
            interpolated_dem_length = geogrid.length + 1
        else:
            interpolated_dem_width = geogrid.width
            interpolated_dem_length = geogrid.length
        out_geo_dem_obj = isce3.io.Raster(
            temp_interpolated_dem.name,
            interpolated_dem_width,
            interpolated_dem_length, 1,
            gdal.GDT_Float32, secondary_layer_files_raster_files_format)
    else:
        temp_interpolated_dem = None
        out_geo_dem_obj = None

    # create a NamedTemporaryFile and an ISCE3 Raster object to
    # temporarily hold the mask layer
    if save_mask:
        temp_mask_file = tempfile.NamedTemporaryFile(
                dir=raster_scratch_dir,
                suffix=secondary_layers_file_extension).name
        out_mask_obj = isce3.io.Raster(
            temp_mask_file,
            geogrid.width, geogrid.length, 1,
            gdal.GDT_Byte, secondary_layer_files_raster_files_format)
    else:
        temp_mask_file = None
        out_mask_obj = None

    # geocode rasters
    geo.geocode(radar_grid=radar_grid,
                input_raster=input_raster_obj,
                output_raster=output_raster_obj,
                dem_raster=dem_raster,
                output_mode=output_mode,
                flag_apply_rtc=flag_apply_rtc,
                input_terrain_radiometry=input_terrain_radiometry,
                output_terrain_radiometry=output_terrain_radiometry,
                rtc_min_value_db=rtc_min_value_db,
                rtc_upsampling=rtc_upsampling,
                rtc_algorithm=rtc_algorithm,
                radargrid_nlooks=radar_grid_nlooks,
                out_off_diag_terms=out_off_diag_terms_obj,
                out_geo_nlooks=out_geo_nlooks_obj,
                out_geo_rtc=out_geo_rtc_obj,
                rtc_area_beta_mode=rtc_area_beta_mode_enum,
                out_geo_rtc_gamma0_to_sigma0=
                    out_geo_rtc_gamma0_to_sigma0_obj,
                out_mask=out_mask_obj,
                input_rtc=None,
                output_rtc=None,
                apply_valid_samples_sub_swath_masking=
                    apply_valid_samples_sub_swath_masking,
                dem_interp_method=dem_interp_method_enum,
                memory_mode=memory_mode,
                **optional_geo_kwargs)

    # delete Raster objects so their associated data is flushed to the disk
    del input_raster_obj
    del output_raster_obj

    if save_nlooks:
        del out_geo_nlooks_obj

    if save_rtc_anf:
        del out_geo_rtc_obj

    if save_rtc_anf_gamma0_to_sigma0:
        del out_geo_rtc_gamma0_to_sigma0_obj

    if save_mask:
        out_mask_obj.close_dataset()
        del out_mask_obj

    if save_dem:
        del out_geo_dem_obj

    if flag_fullcovariance:
        # out_off_diag_terms_obj.close_dataset()
        del out_off_diag_terms_obj

    # For the GCOV workflow, `pol_list` will always be populated. For static layers
    # generation, `pol_list` will be empty.
    if pol_list:
        h5_ds = os.path.join(root_ds, 'listOfPolarizations')
        if h5_ds in hdf5_obj:
            del hdf5_obj[h5_ds]

        pol_list_s2 = np.array(pol_list, dtype='S2')
        dset = hdf5_obj.create_dataset(h5_ds, data=pol_list_s2)
        dset.attrs['description'] = np.bytes_(
            'List of processed polarization layers with frequency ' +
            frequency)

    # save GCOV diagonal elements
    yds, xds = set_get_geo_info(hdf5_obj, root_ds, geogrid)
    cov_elements_list = [p.upper()+p.upper() for p in pol_list]

    # save GCOV imagery
    # `input_raster_list` is optional and not used in static layers generation.
    if input_raster_list:
        save_dataset(temp_output.name, hdf5_obj, root_ds,
                     yds, xds, cov_elements_list,
                     **output_gcov_terms_kwargs)

    # save listOfCovarianceTerms
    freq_group = hdf5_obj[root_ds]
    if not flag_fullcovariance:
        _save_list_cov_terms(cov_elements_list, freq_group)

    # save nlooks
    if save_nlooks:
        save_dataset(temp_nlooks.name, hdf5_obj, root_ds,
                     yds, xds, 'numberOfLooks',
                     **output_secondary_layers_kwargs)

    # save mask
    if save_mask:
        save_dataset(temp_mask_file,
                     hdf5_obj, root_ds,
                     yds, xds,
                     'mask',
                     compute_stats=False)

    # save rtc
    if save_rtc_anf:
        save_dataset(temp_rtc_anf.name, hdf5_obj, root_ds,
                     yds, xds,
                     'rtcAreaNormalizationFactor',
                     **output_secondary_layers_kwargs)

    # save rtc
    if save_rtc_anf_gamma0_to_sigma0:
        save_dataset(temp_rtc_anf_gamma0_to_sigma0.name,
                     hdf5_obj, root_ds,
                     yds, xds,
                     'rtcGammaToSigmaFactor',
                     **output_secondary_layers_kwargs)

    # save interpolated DEM
    if save_dem:

        '''
        The DEM is interpolated over the geogrid pixels vertices
        rather than the pixels centers.
        '''
        if (output_mode ==
                isce3.geocode.GeocodeOutputMode.AREA_PROJECTION):
            dem_geogrid = isce3.product.GeoGridParameters(
                start_x=geogrid.start_x - geogrid.spacing_x / 2,
                start_y=geogrid.start_y - geogrid.spacing_y / 2,
                spacing_x=geogrid.spacing_x,
                spacing_y=geogrid.spacing_y,
                width=int(geogrid.width) + 1,
                length=int(geogrid.length) + 1,
                epsg=geogrid.epsg)
            yds_dem, xds_dem = \
                set_get_geo_info(hdf5_obj, root_ds, dem_geogrid)
        else:
            yds_dem = yds
            xds_dem = xds

        save_dataset(temp_interpolated_dem.name, hdf5_obj,
                     root_ds, yds_dem, xds_dem,
                     'interpolatedDem',
                     long_name='Interpolated DEM',
                     units='1',
                     **output_secondary_layers_kwargs)

    # save GCOV off-diagonal elements
    if flag_fullcovariance:
        off_diag_terms_list = []
        for b1, p1 in enumerate(pol_list):
            for b2, p2 in enumerate(pol_list):
                if (b2 <= b1):
                    continue
                off_diag_terms_list.append(p1.upper()+p2.upper())
        _save_list_cov_terms(cov_elements_list + off_diag_terms_list,
                             freq_group)

        # if the complex data type has not been defined yet,
        # define it. This is required to open the H5 dataset
        # using the netCDF driver
        if '/complex64' not in hdf5_obj:
            complex_type = h5py.h5t.py_create(np.complex64)
            complex_type.commit(hdf5_obj['/'].id,
                                np.bytes_('complex64'))
        else:
            complex_type = hdf5_obj['/complex64']

        save_dataset(temp_off_diag.name, hdf5_obj, root_ds,
                     yds, xds, off_diag_terms_list,
                     long_name=output_radiometry_str,
                     hdf5_data_type=complex_type,
                     **output_gcov_terms_kwargs)


def read_and_validate_rtc_anf_flags(geocode_dict, flag_apply_rtc,
                                    output_terrain_radiometry):
    '''
    Read and validate radiometric terrain correction (RTC) area
    normalization factor (ANF) flags

    Parameters
    ----------
    geocode_dict: dict
        Runconfig geocode namespace
    flag_apply_rtc: bool
        Flag apply RTC (radiometric terrain correction)
    output_terrain_radiometry: isce3.geometry.RtcOutputTerrainRadiometry
        Output terrain radiometry (backscatter coefficient convention)

    Returns
    -------
    save_rtc_anf: bool
        Flag indicating whether the radiometric terrain correction (RTC)
        area normalization factor (ANF) layer should be created.
        This RTC ANF layer provides the conversion factor from
        from gamma0 backscatter normalization convention
        to input backscatter normalization convention
        (e.g., beta0 or sigma0-ellipsoid)
    save_rtc_anf_gamma0_to_sigma0: bool
        Flag indicating whether the radiometric terrain correction (RTC)
        area normalization factor (ANF) gamma0 to sigma0 layer should be
        created
    '''

    info_channel = journal.info("gcov.read_and_validate_rtc_anf_flags")
    error_channel = journal.error("gcov.read_and_validate_rtc_anf_flags")

    save_rtc_anf = geocode_dict['save_rtc_anf']
    save_rtc_anf_gamma0_to_sigma0 = \
        geocode_dict['save_rtc_anf_gamma0_to_sigma0']

    # Verify `flag save_rtc_anf_gamma0_to_sigma0`. The flag defaults to `True`,
    # if `flag_apply_rtc` is enabled and RTC output_type is set to "gamma0", or
    # `False`, otherwise.
    if save_rtc_anf_gamma0_to_sigma0 is None:

        save_rtc_anf_gamma0_to_sigma0 = \
            (flag_apply_rtc and
             output_terrain_radiometry ==
             isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT)

    if not flag_apply_rtc and save_rtc_anf:
        error_msg = (
            "the option `save_rtc_anf` is not available"
            " with radiometric terrain correction"
            " disabled (`apply_rtc = False`).")
        error_channel.log(error_msg)
        raise ValueError(error_msg)

    if not flag_apply_rtc and save_rtc_anf_gamma0_to_sigma0:
        error_msg = (
            "the option `save_rtc_anf_gamma0_to_sigma0`"
            " is not available with radiometric terrain"
            " correction disabled (`apply_rtc = False`).")
        error_channel.log(error_msg)
        raise ValueError(error_msg)

    return save_rtc_anf, save_rtc_anf_gamma0_to_sigma0


class GcovWriter(BaseL2WriterSingleInput):
    """
    Base writer class for NISAR GCOV products
    """

    def __init__(self, runconfig, *args, **kwargs):

        super().__init__(runconfig, *args, **kwargs)

        self.input_freq_pols_dict = self.cfg['processing']['input_subset'][
            'list_of_frequencies']

        # populate the granule ID
        self.get_granule_id(self.input_freq_pols_dict)

        # For GCOV, the input list of polarizations may be different
        # from the output list of polarizations due to the
        # polarimetric symmetrization
        self.freq_pols_dict = self.input_freq_pols_dict.copy()
        flag_symmetrize = self.cfg['processing']['input_subset'][
            'symmetrize_cross_pol_channels']
        if flag_symmetrize:
            for frequency, input_pol_list in self.input_freq_pols_dict.items():

                # if the polarimetric symmetrization is enabled and both
                # cross-pol channels are present, remove channel "VH"
                if 'HV' in input_pol_list and 'VH' in input_pol_list:
                    self.freq_pols_dict[frequency].remove('VH')

    def populate_metadata(self):
        """
        Main method. It calls all other methods
        to populate the product's metadata.
        """
        self.populate_identification_common()
        self.populate_identification_l2_specific()
        self.populate_data_parameters()
        self.populate_calibration_information()
        self.populate_source_data()
        self.populate_processing_information_l2_common()
        self.populate_processing_information()
        self.populate_orbit()
        self.populate_orbit_gcov_specific()
        self.populate_attitude()

        # parse XML specs file
        specs_xml_file = (f'{helpers.WORKFLOW_SCRIPTS_DIR}/'
                          '../products/XML/L2/nisar_L2_GCOV.xml')

        self.check_and_decorate_product_using_specs_xml(specs_xml_file)

    def populate_data_parameters(self):
        """
        Populate the data group `grids` of the GCOV product
        """
        for frequency in self.freq_pols_dict.keys():
            self.copy_from_input(
                '{PRODUCT}/grids/'
                f'frequency{frequency}/numberOfSubSwaths',
                '{PRODUCT}/swaths/'
                f'frequency{frequency}/numberOfSubSwaths',
                skip_if_not_present=True)

    def populate_processing_information(self):
        """
        Populate the `processingInformation` group of the GCOV product
        """

        # populate processing information parameters
        parameters_group = \
            '{PRODUCT}/metadata/processingInformation/parameters'

        # TODO review this
        self.set_value(
            f'{parameters_group}/noiseCorrectionApplied',
            True)

        self.set_value(
            f'{parameters_group}/preprocessingMultilookingApplied',
            False)

        self.set_value(
            f'{parameters_group}/polarizationOrientationCorrectionApplied',
            False)

        self.set_value(
            f'{parameters_group}/faradayRotationApplied',
            False)

        self.copy_from_runconfig(
            f'{parameters_group}/radiometricTerrainCorrectionApplied',
            'processing/geocode/apply_rtc')

        # TODO: read these values from the RSLC metadata once they are
        # available (the RSLC datasets below are not in the specs)
        self.copy_from_input(
            f'{parameters_group}/dryTroposphericGeolocationCorrectionApplied',
            default=True)

        self.copy_from_input(
            f'{parameters_group}/wetTroposphericGeolocationCorrectionApplied',
            default=False)

        self.copy_from_runconfig(
            f'{parameters_group}/rangeIonosphericGeolocationCorrectionApplied',
            'processing/geocode/apply_range_ionospheric_delay_correction')

        self.copy_from_runconfig(
            f'{parameters_group}/'
            'azimuthIonosphericGeolocationCorrectionApplied',
            'processing/geocode/apply_azimuth_ionospheric_delay_correction')

        self.set_value(
            f'{parameters_group}/postProcessingFilteringApplied',
            False)

        self.copy_from_runconfig(
            f'{parameters_group}/isFullCovariance',
            'processing/input_subset/fullcovariance')

        self.copy_from_runconfig(
            f'{parameters_group}/validSamplesSubSwathMaskingApplied',
            'processing/geocode/apply_valid_samples_sub_swath_masking')

        self.copy_from_runconfig(
            f'{parameters_group}/shadowMaskingApplied',
            'processing/geocode/apply_shadow_masking')

        self.copy_from_runconfig(
            f'{parameters_group}/polarimetricSymmetrizationApplied',
            'processing/input_subset/symmetrize_cross_pol_channels')

        # Populate algorithms parameters

        # `run_config_path` can be either a file name or a string
        # representing the contents of the runconfig file (identified
        # by the presence of a "/n" in the `run_config_path`)
        if '\n' in self.runconfig.args.run_config_path:
            try:
                self.set_value(
                    '{PRODUCT}/metadata/processingInformation/parameters/'
                    'runConfigurationContents',
                    self.runconfig.args.run_config_path)
            except UnicodeEncodeError:
                self.set_value(
                    '{PRODUCT}/metadata/processingInformation/parameters/'
                    'runConfigurationContents',
                    self.runconfig.args.run_config_path.encode("utf-8"))
        else:
            with open(self.runconfig.args.run_config_path, "r") as f:
                self.set_value(
                    '{PRODUCT}/metadata/processingInformation/parameters/'
                    'runConfigurationContents',
                    f.read())

        self.copy_from_runconfig(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'demInterpolation',
            'processing/dem_interpolation_method')

        # Add geocoding algorithm reference
        geocoding_algorithm = self.cfg['processing']['geocode'][
            'algorithm_type']
        if geocoding_algorithm == 'area_projection':
            geocoding_algorithm_name = ('Area-Based SAR Geocoding with'
                                        ' Adaptive Multilooking (GEO-AP)')
        else:
            geocoding_algorithm_name = geocoding_algorithm

        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'geocoding',
            geocoding_algorithm_name)

        # Add RTC algorithm reference
        rtc_algorithm = self.cfg['processing']['rtc'][
            'algorithm_type']
        if rtc_algorithm == 'area_projection':
            rtc_algorithm_name = ('Area-Based SAR Radiometric Terrain'
                                  ' Correction (RTC-AP)')
        else:
            rtc_algorithm_name = rtc_algorithm

        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'radiometricTerrainCorrection',
            rtc_algorithm_name)

        input_pol_list = list(self.input_freq_pols_dict.keys())
        flag_hv_and_vh_in_pol_list = ['HV' in input_pol_list and
                                      'VH' in input_pol_list]

        flag_symmetrize = (flag_hv_and_vh_in_pol_list and
                           self.cfg['processing']['input_subset'][
                            'symmetrize_cross_pol_channels'])

        flag_full_covariance = self.cfg['processing']['input_subset'][
            'fullcovariance']

        if flag_symmetrize and not flag_full_covariance:
            symmetrization_algorithm = \
                ('Cross-Polarimetric Channels HV and VH Backscatter Average'
                 ' (Incoherent Average)')
        elif flag_symmetrize:
            symmetrization_algorithm = \
                ('Cross-Polarimetric Channels HV and VH SLCs Average'
                 ' (Coherent Average)')
        else:
            symmetrization_algorithm = 'disabled'

        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'polarimetricSymmetrization',
            symmetrization_algorithm)

        apply_rtc = self.cfg['processing']['geocode']['apply_rtc']
        rtc_algorithm_reference = '(RTC not applied)'
        if apply_rtc:
            if rtc_algorithm == 'area_projection':
                rtc_algorithm_reference = \
                    ('Gustavo H. X. Shiroma, Marco Lavalle, and Sean M.'
                     ' Buckley, "An Area-Based Projection Algorithm for SAR'
                     ' Radiometric Terrain Correction and Geocoding," in IEEE'
                     ' Transactions on Geoscience and Remote Sensing, vol. 60,'
                     ' pp. 1-23, 2022, Art no. 5222723, doi:'
                     ' 10.1109/TGRS.2022.3147472.')
            elif (rtc_algorithm == 'bilinear_distribution'):
                rtc_algorithm_reference = \
                    ('David Small, "Flattening Gamma: Radiometric Terrain'
                     ' Correction for SAR Imagery," in IEEE Transactions on'
                     ' Geoscience and Remote Sensing, vol. 49, no. 8, pp.'
                     ' 3081-3093, Aug. 2011, doi: 10.1109/TGRS.2011.2120616.')
            else:
                error_msg = (f'Unknown RTC algorithm given: {rtc_algorithm}.'
                             ' Supported algorithms: "area_projection",'
                             ' "bilinear_distribution".')
                raise ValueError(error_msg)

        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'radiometricTerrainCorrectionAlgorithmReference',
            rtc_algorithm_reference)

        # TODO: add references to the other geocoding algorithms
        if geocoding_algorithm == 'area_projection':
            geocoding_algorithm_reference = \
                ('Gustavo H. X. Shiroma, Marco Lavalle, and Sean M. Buckley,'
                 ' "An Area-Based Projection Algorithm for SAR Radiometric'
                 ' Terrain Correction and Geocoding," in IEEE Transactions'
                 ' on Geoscience and Remote Sensing, vol. 60, pp. 1-23, 2022,'
                 ' Art no. 5222723, doi: 10.1109/TGRS.2022.3147472.')
        else:
            geocoding_algorithm_reference = '(NOT SPECIFIED)'

        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'geocodingAlgorithmReference',
            geocoding_algorithm_reference)

        # populate preprocessing parameters
        for frequency, _ in self.freq_pols_dict.items():
            preprocessing_group_path = \
                f'{parameters_group}/preprocessing/frequency{frequency}/'
            self.copy_from_runconfig(
                f'{preprocessing_group_path}/numberOfRangeLooks',
                'processing/pre_process/range_looks',
                format_function=np.uint64)
            self.copy_from_runconfig(
                f'{preprocessing_group_path}/numberOfAzimuthLooks',
                'processing/pre_process/azimuth_looks',
                format_function=np.uint64)

        # populate rtc parameters
        self.copy_from_runconfig(
            f'{parameters_group}/rtc/inputBackscatterNormalizationConvention',
            'processing/rtc/input_terrain_radiometry')

        if apply_rtc:
            self.copy_from_runconfig(
                f'{parameters_group}/rtc/'
                'outputBackscatterNormalizationConvention',
                'processing/rtc/output_type')
        else:
            self.set_value(
                f'{parameters_group}/rtc/'
                'outputBackscatterNormalizationConvention',
                'beta0')

        self.set_value(
            f'{parameters_group}/rtc/'
            'outputBackscatterExpressionConvention',
            'backscatter intensity (linear)')

        self.copy_from_runconfig(
            f'{parameters_group}/rtc/memoryMode',
            'processing/geocode/memory_mode')

        self.copy_from_runconfig(
            f'{parameters_group}/rtc/minRtcAreaNormalizationFactorInDB',
            'processing/rtc/rtc_min_value_db',
            format_function=float)

        self.copy_from_runconfig(
            f'{parameters_group}/rtc/geogridUpsampling',
            'processing/rtc/dem_upsampling',
            format_function=float)

        # populate geocoding parameters
        self.copy_from_runconfig(
            f'{parameters_group}/geocoding/memoryMode',
            'processing/geocode/memory_mode')

        self.copy_from_runconfig(
            f'{parameters_group}/geocoding/geogridUpsampling',
            'processing/geocode/geogrid_upsampling',
            format_function=float)

        self.copy_from_runconfig(
            f'{parameters_group}/geocoding/minBlockSize',
            'processing/geocode/min_block_size',
            default=isce3.core.default_min_block_size,
            format_function=np.uint64)

        self.copy_from_runconfig(
            f'{parameters_group}/geocoding/maxBlockSize',
            'processing/geocode/max_block_size',
            default=isce3.core.default_max_block_size,
            format_function=np.uint64)

        self.copy_from_runconfig(
            f'{parameters_group}/geocoding/isSourceDataUpsampled',
            'processing/geocode/upsample_radargrid')

        # populate geo2rdr parameters
        self.copy_from_runconfig(
            f'{parameters_group}/geo2rdr/convergenceThreshold',
            'processing/geo2rdr/threshold')

        self.copy_from_runconfig(
            f'{parameters_group}/geo2rdr/maxNumberOfIterations',
            'processing/geo2rdr/maxiter',
            format_function=np.uint64)

        # this value is hard-coded in the GeocodeCov module
        self.set_value(
            f'{parameters_group}/geo2rdr/deltaRange',
            1.0e-8)

    def populate_orbit_gcov_specific(self):
        """
        Populate GCOV-specific `orbit` datasets `interpMethod` and
        `referenceEpoch`
        """
        error_channel = journal.error(
            'GcovWriter.populate_orbit_gcov_specific')

        # save the orbit interp method
        interp_method_path = (f'{self.output_product_path}/metadata/orbit/'
                              'interpMethod')
        if interp_method_path not in self.output_hdf5_obj:
            orbit_interp_method = self.orbit.get_interp_method()
            if orbit_interp_method == isce3.core.OrbitInterpMethod.HERMITE:
                orbit_interp_method_str = 'Hermite'
            elif orbit_interp_method == isce3.core.OrbitInterpMethod.LEGENDRE:
                orbit_interp_method_str = 'Legendre'
            else:
                error_msg = "unexpected orbit interpolation method"
                error_channel.log(error_msg)
                raise ValueError(error_msg)
            self.set_value(
                '{PRODUCT}/metadata/orbit/interpMethod',
                orbit_interp_method_str)

        self.set_value(
            '{PRODUCT}/metadata/orbit/referenceEpoch',
            self.orbit.reference_epoch.isoformat_usec())
