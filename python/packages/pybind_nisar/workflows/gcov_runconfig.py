import numpy as np

import journal

import pybind_isce3 as isce
from pybind_nisar.workflows.runconfig import RunConfig

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
        error_channel = journal.error('gcov_runconfig.prep_gcov')

        geocode_dict = self.cfg['processing']['geocode']

        if geocode_dict['abs_rad_cal'] is None:
            geocode_dict['abs_rad_cal'] = 1.0

        if geocode_dict['memory_mode'] == 'single_block':
            geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.SINGLE_BLOCK
        elif geocode_dict['memory_mode'] == 'geogrid':
            geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.BLOCKS_GEOGRID
        elif geocode_dict['memory_mode'] == 'geogrid_radargrid':
            geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.BLOCKS_GEOGRID_AND_RADARGRID
        else:
            geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.AUTO

        if geocode_dict['algorithm_type'] == 'interp':
            geocode_dict['algorithm_type'] = isce.geocode.GeocodeOutputMode.INTERP
        elif geocode_dict['algorithm_type'] == 'area_projection':
            geocode_dict['algorithm_type'] = isce.geocode.GeocodeOutputMode.AREA_PROJECTION
        elif geocode_dict['algorithm_type'] == 'area_projection_gamma_naught':
            geocode_dict['algorithm_type'] = isce.geocode.GeocodeOutputMode.AREA_PROJECTION_GAMMA_NAUGHT
        else:
            err_str = f'Unsupported geocode algorithm: {geocode_dict["algorithm_type"]}'
            error_channel.log(err_str)
            raise ValueError(err_str)

        rtc_dict = self.cfg['processing']['rtc']

        # only 2 RTC algorithms supported: david-small (default) & area-projection
        if rtc_dict['algorithm_type'] == "area_projection":
            rtc_dict['algorithm_type'] = isce.geometry.RtcAlgorithm.RTC_AREA_PROJECTION
        else:
            rtc_dict['algorithm_type'] = isce.geometry.RtcAlgorithm.RTC_DAVID_SMALL

        if rtc_dict['input_terrain_radiometry'] == "sigma0":
            rtc_dict['input_terrain_radiometry'] = isce.geometry.RtcInputRadiometry.SIGMA_NAUGHT_ELLIPSOID
        else:
            rtc_dict['input_terrain_radiometry'] = isce.geometry.RtcInputRadiometry.BETA_NAUGHT

        if rtc_dict['rtc_min_value_db'] is None:
            rtc_dict['rtc_min_value_db'] = np.nan
