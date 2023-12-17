import numpy as np

import isce3
from nisar.products.writers import BaseWriterSingleInput


class BaseL2WriterSingleInput(BaseWriterSingleInput):
    """
    Base L2 writer class that can be used for NISAR L2 products
    """

    def __init__(self, runconfig, *args, **kwargs):

        super().__init__(runconfig, *args, **kwargs)

    def populate_identification_l2_specific(self):
        """
        Populate L2 product specific parameters in the
        identification group
        """

        self.copy_from_input(
            'identification/zeroDopplerStartTime')

        self.copy_from_input(
            'identification/zeroDopplerEndTime')

        self.set_value(
            'identification/productLevel',
            'L2')

        is_geocoded = True
        self.set_value(
            'identification/isGeocoded',
            is_geocoded)

        # TODO populate attribute `epsg`
        self.copy_from_input(
            'identification/boundingPolygon')

        self.set_value(
            'identification/listOfFrequencies',
            list(self.freq_pols_dict.keys()))

    def populate_calibration_information(self):

        # calibration parameters to be copied from the RSLC
        # common to all polarizations
        calibration_freq_parameter_list = ['commonDelay',
                                           'faradayRotation']

        # calibration parameters to be copied from the RSLC
        # specific to each polarization
        calibration_freq_pol_parameter_list = \
            ['differentialDelay',
             'differentialPhase',
             'scaleFactor',
             'scaleFactorSlope']

        for frequency in self.freq_pols_dict.keys():

            for parameter in calibration_freq_parameter_list:
                cal_freq_path = (
                    '{PRODUCT}/metadata/calibrationInformation/'
                    f'frequency{frequency}')

                self.copy_from_input(f'{cal_freq_path}/{parameter}',
                                     default=np.nan)

            # The following parameters are available for all
            # quad pols in the lexicographic base
            # regardless of listOfPolarizations
            for pol in ['HH', 'HV', 'VH', 'VV']:
                for parameter in calibration_freq_pol_parameter_list:
                    self.copy_from_input(f'{cal_freq_path}/{pol}/{parameter}',
                                         default=np.nan)

    def populate_orbit(self):
        # RSLC products before v0.9.0 may include "acceleration",
        # that should not be copied to L2 products
        excludes_list = ['acceleration']

        # copy orbit information group
        self._copy_group_from_input('{PRODUCT}/metadata/orbit',
                                    excludes=excludes_list)

    def populate_attitude(self):
        # RSLC products before v0.9.0 may include "angularVelocity".
        # that should not be copied to L2 products
        excludes_list = ['angularVelocity']

        # copy attitude information group
        self._copy_group_from_input('{PRODUCT}/metadata/attitude',
                                    excludes=excludes_list)

    def populate_processing_information_l2_common(self):
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
        if '\n' not in self.runconfig.args.run_config_path:
            self.set_value(
                f'{inputs_group}/configFiles',
                [self.runconfig.args.run_config_path])

        self.copy_from_runconfig(
            f'{inputs_group}/demSource',
            'dynamic_ancillary_file_group/dem_file_description',
            default='(NOT SPECIFIED)')
