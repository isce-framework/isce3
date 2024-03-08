import journal
import isce3

import nisar.workflows.helpers as helpers
from nisar.products.writers import BaseL2WriterSingleInput


class GcovWriter(BaseL2WriterSingleInput):
    """
    Base writer class for NISAR GCOV products
    """

    def __init__(self, runconfig, *args, **kwargs):

        super().__init__(runconfig, *args, **kwargs)

        self.input_freq_pols_dict = self.cfg['processing']['input_subset'][
            'list_of_frequencies']

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

    def populate_source_data(self):
        """
        Populate the `sourceData` group of the GCOV product
        """

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/productVersion',
            'identification/productVersion',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/lookDirection',
            'identification/lookDirection',
            format_function=str.title)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/productLevel',
            'identification/productLevel',
            default='L1')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingDateTime',
            'identification/processingDateTime',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'parameters/runConfigurationContents',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/rfiDetection',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rfiDetection',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/rfiMitigation',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rfiMitigation',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/rangeCompression',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rangeCompression',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/elevationAntennaPatternCorrection',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'elevationAntennaPatternCorrection',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/rangeSpreadingLossCorrection',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rangeSpreadingLossCorrection',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/dopplerCentroidEstimation',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'dopplerCentroidEstimation',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/azimuthPresumming',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'azimuthPresumming',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/azimuthCompression',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'azimuthCompression',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/softwareVersion',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'softwareVersion',
            skip_if_not_present=True)

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

        self.copy_from_input(
            f'{parameters_group}/rfiCorrectionApplied',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rfiMitigation',
            default=False)

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

    def populate_orbit_gcov_specific(self):
        """
        Populate GCOV-specific `orbit` datasets `interpMethod` and
        `referenceEpoch`
        """
        # TODO: update the code below once the capability of storing an
        # external orbit file is implemented

        error_channel = journal.error(
            'GcovWriter.populate_orbit_gcov_specific')
        rslc_orbit_path = f'{self.output_product_path}/metadata/orbit'
        orbit = isce3.core.load_orbit_from_h5_group(
            self.output_hdf5_obj[rslc_orbit_path])

        # The orbit `interpMethod`` was removed from the RSLC product
        # specification in ISCE3 Release 4. However, RSLC products
        # with previous version may include the dataset. If the field
        # is not present in the GCOV orbit group, i.e., if it was not
        # copied from the RSLC orbit
        # group, add it.
        input_orbit_group_path = \
            (f'{self.root_path}/{self.input_product_hdf5_group_type}'
             '/metadata/orbit/interpMethod')

        if input_orbit_group_path not in self.input_hdf5_obj:
            orbit_interp_method = orbit.get_interp_method()
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
            orbit.reference_epoch.isoformat_usec())
