import journal
import numpy as np

import isce3
from nisar.workflows.runconfig import RunConfig
from nisar.products.readers import SLC


class GCOVRunConfig(RunConfig):
    def __init__(self, args):
        super().__init__(args, 'gcov')
        super().load_geocode_yaml_to_dict()
        super().geocode_common_arg_load()
        self.load()

    def load(self):
        '''
        Load GCOV specific parameters.
        '''
        warning_channel = journal.warning('GCOVRunConfig.load()')

        flag_fullcovariance = self.cfg['processing']['input_subset'][
            'fullcovariance']

        # Handle the case in which the `fullcovariance` flag field is empty.
        if flag_fullcovariance is None:

            # The choice on whether to process in full-covariance mode
            # or not will depend on the list of frequency and polarizations
            # to process. This list has already been been verified based
            # on the runconfig and available RSLC polarimetric channels
            freq_pols_dict = self.cfg['processing']['input_subset'][
                'list_of_frequencies']

            # `fullcovariance` is set to `True` if any frequency
            # to be processed includes a full-pol dataset,
            # i.e., includes both co-pols HH and VV
            # and at least one cross-pol HV or VH.
            flag_fullcovariance = False
            for freq, pol_list in freq_pols_dict.items():

                # Verify if frequency to process is full-pol. It's considered
                # full-pol if it contains both co-pols and at least one
                # cross-pol.
                if (('HH' in pol_list) and ('VV' in pol_list) and
                   ('HV' in pol_list or 'VH' in pol_list)):
                    warning_channel.log(
                        'The `fullcovariance` field is not specified in the'
                        ' runconfig. By default, it is set to `True` if any'
                        ' processed frequency includes a full-pol dataset,'
                        ' and `False` otherwise.'
                        f' Since frequency {freq} contains polarizations'
                        f' {pol_list}, `fullcovariance` is set to `True`.')
                    flag_fullcovariance = True
                    break
            else:
                warning_channel.log(
                    'The `fullcovariance` field is not specified in the'
                    ' runconfig. By default, it is set to `True` if any'
                    ' processed frequency includes a full-pol dataset,'
                    ' and `False` otherwise.'
                    ' Since the output product will not contain a full-'
                    'polarimetric dataset, `fullcovariance` is set to'
                    ' `False`.')

            self.cfg['processing']['input_subset']['fullcovariance'] = \
                flag_fullcovariance

        geocode_dict = self.cfg['processing']['geocode']
        rtc_dict = self.cfg['processing']['rtc']

        tec_file = self.cfg["dynamic_ancillary_file_group"]['tec_file']

        if geocode_dict['apply_range_ionospheric_delay_correction'] is None:
            geocode_dict['apply_range_ionospheric_delay_correction'] = \
                tec_file is not None

        if geocode_dict['apply_azimuth_ionospheric_delay_correction'] is None:
            geocode_dict['apply_azimuth_ionospheric_delay_correction'] = \
                tec_file is not None

        if geocode_dict['abs_rad_cal'] is None:
            geocode_dict['abs_rad_cal'] = 1.0

        if geocode_dict['clip_max'] is None:
            geocode_dict['clip_max'] = np.nan

        if geocode_dict['clip_min'] is None:
            geocode_dict['clip_min'] = np.nan

        if geocode_dict['geogrid_upsampling'] is None:
            geocode_dict['geogrid_upsampling'] = 1.0

        geocode_dict['memory_mode_enum'] = \
            isce3.core.normalize_geocode_memory_mode(geocode_dict['memory_mode'])

        rtc_output_type = rtc_dict['output_type']
        if rtc_output_type == 'sigma0':
            rtc_dict['output_type_enum'] = \
                isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT
        else:
            rtc_dict['output_type_enum'] = \
                isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT

        geocode_algorithm = self.cfg['processing']['geocode']['algorithm_type']
        geocode_dict['output_mode'] = \
            isce3.geocode.normalize_geocode_output_mode(geocode_algorithm)

        # only 2 RTC algorithms supported: area_projection (default) &
        # bilinear_distribution
        rtc_dict['algorithm_type_enum'] = \
            isce3.geometry.normalize_rtc_algorithm(rtc_dict['algorithm_type'])

        if rtc_dict['input_terrain_radiometry'] == "sigma0":
            rtc_dict['input_terrain_radiometry_enum'] = \
                isce3.geometry.RtcInputTerrainRadiometry.SIGMA_NAUGHT_ELLIPSOID
        else:
            rtc_dict['input_terrain_radiometry_enum'] = \
                isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT

        if rtc_dict['rtc_min_value_db'] is None:
            rtc_dict['rtc_min_value_db'] = np.nan

        # Update the DEM interpolation method
        dem_interp_method = self.cfg['processing']['dem_interpolation_method']
        self.cfg['processing']['dem_interpolation_method_enum'] = \
            isce3.core.normalize_data_interp_method(dem_interp_method)
