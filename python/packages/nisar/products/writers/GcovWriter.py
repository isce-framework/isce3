import isce3
from nisar.products.writers import BaseL2WriterSingleInput


class GcovWriter(BaseL2WriterSingleInput):
    """
    Base writer class for NISAR GCOV products
    """

    def __init__(self, runconfig, *args, **kwargs):

        super().__init__(runconfig, *args, **kwargs)

        self.input_freq_pols_dict = self.cfg['processing']['input_subset'][
            'list_of_frequencies']

        self.output_format = runconfig.cfg['output_gcov_terms']['format']

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
        self.populate_processing_information()
        self.populate_orbit()
        self.populate_attitude()

    def populate_data_parameters(self):
        for frequency, _ in self.freq_pols_dict.items():
            self.copy_from_input(
                '{PRODUCT}/grids/'
                f'frequency{frequency}/numberOfSubSwaths',
                '{PRODUCT}/swaths/'
                f'frequency{frequency}/numberOfSubSwaths',
                default='(NOT SPECIFIED)')

    def populate_source_data(self):

        try:
            self.copy_from_input(
                '{PRODUCT}/metadata/sourceData/productVersion',
                'identification/productVersion')
        except KeyError:
            self.set_value(
                    '{PRODUCT}/metadata/sourceData/productVersion',
                    '(NOT SPECIFIED)')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/lookDirection',
            'identification/lookDirection',
            format_function=str.title)

        # TODO: remove this `try/except` once `productLevel`
        # is implemented for all input products (e.g., RSLC)
        try:
            self.copy_from_input(
                '{PRODUCT}/metadata/sourceData/productLevel',
                'identification/productLevel')
        except KeyError:
            self.set_value(
                '{PRODUCT}/metadata/sourceData/productLevel',
                'L1')

        # TODO: remove this `try/except` once `processingDateTime`
        # is implemented for all input products (e.g., RSLC)
        try:
            self.copy_from_input(
                '{PRODUCT}/metadata/sourceData/processingDateTime',
                'identification/processingDateTime')
        except KeyError:
            self.set_value(
                '{PRODUCT}/metadata/sourceData/processingDateTime',
                '(NOT SPECIFIED)')

        # TODO: remove this `try/except` once `processingDateTime`
        # is implemented for all input products (e.g., RSLC)
        try:
            self.copy_from_input(
                '{PRODUCT}/metadata/sourceData/processingInformation/'
                'parameters/runConfigurationContents',
                '{PRODUCT}/metadata/processingInformation/parameters/'
                'runConfigurationContents')
        except KeyError:
            self.set_value(
                '{PRODUCT}/metadata/sourceData/processingInformation/'
                'parameters/runConfigurationContents',
                '(NOT SPECIFIED)')

        # TODO: remove this `try/except` once `rfiDetection`
        # is implemented for all input products (e.g., RSLC)
        try:
            self.copy_from_input(
                '{PRODUCT}/metadata/sourceData/processingInformation/'
                'algorithms/rfiDetection',
                '{PRODUCT}/metadata/processingInformation/algorithms/'
                'rfiDetection')
        except KeyError:
            self.set_value(
                '{PRODUCT}/metadata/sourceData/processingInformation/'
                'algorithms/rfiDetection',
                '(NOT SPECIFIED)')

        # TODO: remove this `try/except` once `rfiMitigation`
        # is implemented for all input products (e.g., RSLC)
        try:
            self.copy_from_input(
                '{PRODUCT}/metadata/sourceData/processingInformation/'
                'algorithms/rfiMitigation',
                '{PRODUCT}/metadata/processingInformation/algorithms/'
                'rfiMitigation')
        except KeyError:
            self.set_value(
                '{PRODUCT}/metadata/sourceData/processingInformation/'
                'algorithms/rfiMitigation',
                '(NOT SPECIFIED)')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/rangeCompression',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rangeCompression')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/elevationAntennaPatternCorrection',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'elevationAntennaPatternCorrection')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/rangeSpreadingLossCorrection',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rangeSpreadingLossCorrection')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/dopplerCentroidEstimation',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'dopplerCentroidEstimation')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/azimuthPresumming',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'azimuthPresumming')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/azimuthCompression',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'azimuthCompression')

        # TODO: remove this `try/except` once `softwareVersion`
        # is implemented for all input products (e.g., RSLC)
        try:
            self.copy_from_input(
                '{PRODUCT}/metadata/sourceData/processingInformation/'
                'algorithms/softwareVersion',
                '{PRODUCT}/metadata/processingInformation/algorithms/'
                'softwareVersion')
        except KeyError:
            self.set_value(
                '{PRODUCT}/metadata/sourceData/processingInformation/'
                'algorithms/softwareVersion',
                '(NOT SPECIFIED)')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/swaths/zeroDopplerStartTime',
            'identification/zeroDopplerStartTime')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/swaths/zeroDopplerTimeSpacing',
            '{PRODUCT}/swaths/zeroDopplerTimeSpacing')

        for i, (frequency, _) in enumerate(self.freq_pols_dict.items()):
            radar_grid_obj = self.input_product_obj.getRadarGrid(frequency)

            output_swaths_freq_path = ('{PRODUCT}/metadata/sourceData/'
                                       f'swaths/frequency{frequency}')
            input_swaths_freq_path = ('{PRODUCT}/swaths/'
                                      f'frequency{frequency}')

            if i == 0:
                self.set_value(
                    '{PRODUCT}/metadata/sourceData/swaths/'
                    'numberOfAzimuthLines',
                    radar_grid_obj.length)

            self.copy_from_input(
                f'{output_swaths_freq_path}/rangeBandwidth',
                f'{input_swaths_freq_path}/processedRangeBandwidth')

            self.copy_from_input(
                f'{output_swaths_freq_path}/azimuthBandwidth',
                f'{input_swaths_freq_path}/processedAzimuthBandwidth')

            self.copy_from_input(
                f'{output_swaths_freq_path}/centerFrequency',
                f'{input_swaths_freq_path}/processedCenterFrequency')

            self.set_value(
                f'{output_swaths_freq_path}/slantRangeStart',
                radar_grid_obj.starting_range)

            self.copy_from_input(
                f'{output_swaths_freq_path}/slantRangeSpacing',
                f'{input_swaths_freq_path}/slantRangeSpacing')

            self.set_value(
                f'{output_swaths_freq_path}/numberOfRangeSamples',
                radar_grid_obj.width)

    def populate_processing_information(self):

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

        self.copy_from_runconfig(
            f'{parameters_group}/dryTroposphericGeolocationCorrectionApplied',
            'processing/geocode/apply_dry_tropospheric_delay_correction')

        self.copy_from_runconfig(
            f'{parameters_group}/wetTroposphericGeolocationCorrectionApplied',
            'processing/geocode/apply_wet_tropospheric_delay_correction')

        self.copy_from_runconfig(
            f'{parameters_group}/rangeIonosphericGeolocationCorrectionApplied',
            'processing/geocode/apply_range_ionospheric_delay_correction')

        self.copy_from_runconfig(
            f'{parameters_group}/'
            'azimuthIonosphericGeolocationCorrectionApplied',
            'processing/geocode/apply_azimuth_ionospheric_delay_correction')

        # TODO: remove this `try/except` once `rfiMitigation`
        # is implemented for all input products (e.g., RSLC)
        try:
            self.copy_from_input(
                f'{parameters_group}/rfiCorrectionApplied',
                '{PRODUCT}/metadata/processingInformation/algorithms/'
                'rfiMitigation')
        except KeyError:
            self.set_value(
                f'{parameters_group}/rfiCorrectionApplied',
                False)

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

        # TODO fix this
        flag_rfi = (f'{self.root_path}/'
                    f'{self.input_product_hdf5_group_type}'
                    '/metadata/processingInformation/algorithms/rfiMitigation')
        if flag_rfi:
            rfi_algorithm_reference = '(RFI correction not applied)'
        else:
            rfi_algorithm_reference = '(NOT SPECIFIED)'
        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rfiCorrection',
            rfi_algorithm_reference)

        # TODO fix this
        flag_symmetrize = self.cfg['processing']['input_subset'][
            'symmetrize_cross_pol_channels']
        if flag_symmetrize:
            symmetrization_algorithm_reference = '(RFI correction not applied)'
        else:
            symmetrization_algorithm_reference = '(NOT SPECIFIED)'
        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'polarimetricSymmetrization',
            symmetrization_algorithm_reference)

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
                'processing/pre_process/range_looks')
            self.copy_from_runconfig(
                f'{preprocessing_group_path}/numberOfAzimuthLooks',
                'processing/pre_process/azimuth_looks')

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
            'processing/rtc/rtc_min_value_db')

        self.copy_from_runconfig(
            f'{parameters_group}/rtc/geogridUpsampling',
            'processing/rtc/dem_upsampling')

        # populate geocoding parameters
        self.copy_from_runconfig(
            f'{parameters_group}/geocoding/memoryMode',
            'processing/geocode/memory_mode')

        self.copy_from_runconfig(
            f'{parameters_group}/geocoding/geogridUpsampling',
            'processing/geocode/geogrid_upsampling')

        self.copy_from_runconfig(
            f'{parameters_group}/geocoding/minBlockSize',
            'processing/geocode/min_block_size',
            default=isce3.core.default_min_block_size)

        self.copy_from_runconfig(
            f'{parameters_group}/geocoding/maxBlockSize',
            'processing/geocode/max_block_size',
            default=isce3.core.default_max_block_size)

        self.copy_from_runconfig(
            f'{parameters_group}/geocoding/isSourceDataUpsampled',
            'processing/geocode/upsample_radargrid')

        # populate geo2rdr parameters
        self.copy_from_runconfig(
            f'{parameters_group}/geo2rdr/convergenceThreshold',
            'processing/geo2rdr/threshold')

        self.copy_from_runconfig(
            f'{parameters_group}/geo2rdr/maxNumberOfIterations',
            'processing/geo2rdr/maxiter')

        # this value is hard-coded in the GeocodeCov module
        self.set_value(
            f'{parameters_group}/geo2rdr/deltaRange',
            1.0e-8)

        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'softwareVersion',
            isce3.__version__)

        # populate processing information inputs parameters
        inputs_group = \
            '{PRODUCT}/metadata/processingInformation/inputs'

        self.set_value(
            f'{inputs_group}/l1SlcGranules',
            [self.input_file])

        orbit_file = self.cfg[
            'dynamic_ancillary_file_group']['orbit_file']
        if orbit_file is None:
            orbit_file = '(NOT SPECIFIED)'
        self.set_value(
            f'{inputs_group}/orbitFiles',
            [orbit_file])

        # `run_config_path` can be either a file name or a string
        # representing the contents of the runconfig file (identified
        # by the presence of a "/n" in the `run_config_path`)
        if '\n' in self.runconfig.args.run_config_path:
            self.set_value(
                f'{inputs_group}/configFiles',
                '(NOT SPECIFIED)')
        else:
            self.set_value(
                f'{inputs_group}/configFiles',
                [self.runconfig.args.run_config_path])

        dem_file_description = \
            self.cfg['dynamic_ancillary_file_group']['dem_file_description']
        if dem_file_description is None:
            dem_file_description = '(NOT SPECIFIED)'
        self.set_value(
            f'{inputs_group}/demSource',
            dem_file_description)
