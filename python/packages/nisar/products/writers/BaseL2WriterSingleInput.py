import numpy as np
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

        for frequency, pol_list in self.freq_pols_dict.items():

            for parameter in calibration_freq_parameter_list:
                cal_freq_path = (
                    '{PRODUCT}/metadata/calibrationInformation/'
                    f'frequency{frequency}')

                self.copy_from_input(f'{cal_freq_path}/{parameter}',
                                     default=np.nan)

            for pol in pol_list:
                for parameter in calibration_freq_pol_parameter_list:
                    self.copy_from_input(f'{cal_freq_path}/{pol}/{parameter}',
                                         default=np.nan)

    def populate_orbit(self):
        # copy orbit information group
        self._copy_group_from_input(
            '{PRODUCT}/metadata/orbit')

    def populate_attitude(self):
        # copy attitude information group
        self._copy_group_from_input(
            '{PRODUCT}/metadata/attitude')
