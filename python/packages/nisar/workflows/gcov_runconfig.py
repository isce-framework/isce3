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

        # Check if the `fullcovariance` flag field is empty.
        # If so, the YAML parser assigns it the string value `"None"`.
        if (flag_fullcovariance is None or
                (isinstance(flag_fullcovariance, str) and
                 flag_fullcovariance == 'None')):

            # If empty, `fullcovariance` is set to `True` any frequency
            # to be processed includes a full-pol dataset
            # (3 or 4 polarizations), and `False` otherwise.
            freq_pols_dict = self.cfg['processing']['input_subset'][
                'list_of_frequencies']

            flag_process_fullpol = False
            for freq, pol_list in freq_pols_dict.items():

                flag_process_fullpol_this_frequency = \
                    (all([pol in pol_list
                         for pol in ['HH', 'VV', 'HV']]) or
                     all([pol in pol_list
                         for pol in ['HH', 'VV', 'VH']]))

                if (not flag_process_fullpol and
                        flag_process_fullpol_this_frequency):
                    warning_channel.log(
                        'The `fullcovariance` flag is empty in the runconfig. '
                        'By default, it is set to `True` if any frequency to'
                        ' be processed includes a full-pol dataset. This is'
                        f' the case for frequency {freq} with polarizations'
                        f' {pol_list}. Setting `fullcovariance` to `True`.')

                flag_process_fullpol |= flag_process_fullpol_this_frequency

            if not flag_fullcovariance:
                warning_channel.log(
                    'The `fullcovariance` flag is empty in the runconfig. '
                    'By default, it is set to `True` if any frequency to be'
                    ' processed includes a full-pol dataset, which is not the '
                    'case for the given input RSLC and runconfig. '
                    'Setting `fullcovariance` to `False`.')

            self.cfg['processing']['input_subset']['fullcovariance'] = \
                flag_process_fullpol

            flag_fullcovariance = self.cfg['processing']['input_subset'][
                'fullcovariance']

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
