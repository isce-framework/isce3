from nisar.products.writers import BaseL2WriterSingleInput


class GslcWriter(BaseL2WriterSingleInput):
    """
    Base writer class for NISAR GSLC products
    """

    def __init__(self, runconfig, *args, **kwargs):

        super().__init__(runconfig, *args, **kwargs)

        self.freq_pols_dict = self.cfg['processing']['input_subset'][
            'list_of_frequencies']

    def populate_metadata(self):
        """
        Main method. It calls all other methods
        to populate the product's metadata.
        """
        self.populate_identification_common()
        self.populate_identification_l2_specific()
        self.populate_data_parameters()
        self.populate_calibration_information()
        self.populate_processing_information_l2_common()
        self.populate_processing_information()
        self.populate_orbit()
        self.populate_attitude()

    def populate_data_parameters(self):

        for frequency, _ in self.freq_pols_dict.items():
            input_swaths_freq_path = ('{PRODUCT}/swaths/'
                                      f'frequency{frequency}')
            output_swaths_freq_path = ('{PRODUCT}/grids/'
                                       f'frequency{frequency}')
            self.copy_from_input(
                f'{output_swaths_freq_path}/numberOfSubSwaths',
                f'{input_swaths_freq_path}/numberOfSubSwaths',
                skip_if_not_present=True)

            self.copy_from_input(
                f'{output_swaths_freq_path}/rangeBandwidth',
                f'{input_swaths_freq_path}/processedRangeBandwidth')

            self.copy_from_input(
                f'{output_swaths_freq_path}/azimuthBandwidth',
                f'{input_swaths_freq_path}/processedAzimuthBandwidth')

            self.copy_from_input(
                f'{output_swaths_freq_path}/centerFrequency',
                f'{input_swaths_freq_path}/processedCenterFrequency')

            self.copy_from_input(
                f'{output_swaths_freq_path}/slantRangeSpacing',
                f'{input_swaths_freq_path}/slantRangeSpacing')

            self.copy_from_input(
                f'{output_swaths_freq_path}/zeroDopplerTimeSpacing',
                '{PRODUCT}/swaths/zeroDopplerTimeSpacing')

    def populate_processing_information(self):

        # populate processing information parameters
        parameters_group = \
            '{PRODUCT}/metadata/processingInformation/parameters'

        self.copy_from_input(
            f'{parameters_group}/azimuthChirpWeighting',
            skip_if_not_present=True)
        self.copy_from_input(
            f'{parameters_group}/rangeChirpWeighting',
            skip_if_not_present=True)

        # TODO: verify values below
        self.set_value(
            f'{parameters_group}/dryTroposphericGeolocationCorrectionApplied',
            True)

        self.set_value(
            f'{parameters_group}/wetTroposphericGeolocationCorrectionApplied',
            False)

        tec_file = self.cfg["dynamic_ancillary_file_group"]['tec_file']
        flag_ionopheric_correction_enabled = tec_file is not None
        self.set_value(
            f'{parameters_group}/rangeIonosphericGeolocationCorrectionApplied',
            flag_ionopheric_correction_enabled)

        # TODO: verify
        self.set_value(
            f'{parameters_group}/'
            'azimuthIonosphericGeolocationCorrectionApplied',
            flag_ionopheric_correction_enabled)

        self.copy_from_input(
            f'{parameters_group}/rfiCorrectionApplied',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rfiMitigation',
            default=False)

        self.copy_from_runconfig(
            f'{parameters_group}/ellipsoidalFlatteningApplied',
            'processing/flatten')

        # TODO: verify
        self.set_value(
            f'{parameters_group}/topographicFlatteningApplied',
            True)

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

        # TODO: verify (really hard-coded to bilinear???)
        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'demInterpolation',
            'bilinear')

        # Geocoding algorithm
        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'geocoding',
            'Sinc interpolation')
