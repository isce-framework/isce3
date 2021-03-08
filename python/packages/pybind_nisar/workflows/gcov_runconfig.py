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
        error_channel = journal.error('gcov_runconfig.load')

        geocode_dict = self.cfg['processing']['geocode']
        rtc_dict = self.cfg['processing']['rtc']

        if geocode_dict['abs_rad_cal'] is None:
            geocode_dict['abs_rad_cal'] = 1.0

        if geocode_dict['clip_max'] is None:
            geocode_dict['clip_max'] = 5.0

        if geocode_dict['clip_min'] is None:
            geocode_dict['clip_min'] = 0.0

        if geocode_dict['geogrid_upsampling'] is None:
            geocode_dict['geogrid_upsampling'] = 1.0

        if geocode_dict['memory_mode'] == 'single_block':
            geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.SINGLE_BLOCK
        elif geocode_dict['memory_mode'] == 'geogrid':
            geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.BLOCKS_GEOGRID
        elif geocode_dict['memory_mode'] == 'geogrid_radargrid':
            geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.BLOCKS_GEOGRID_AND_RADARGRID
        else:
            geocode_dict['memory_mode'] = isce.geocode.GeocodeMemoryMode.AUTO

        rtc_output_type = rtc_dict['output_type']
        if rtc_output_type == 'sigma0':
            rtc_dict['output_type'] = isce.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT
        else:
            rtc_dict['output_type'] = isce.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT

        if geocode_dict['algorithm_type'] == 'interp':
            geocode_dict['algorithm_type'] = isce.geocode.GeocodeOutputMode.INTERP
        else:
            geocode_dict['algorithm_type'] = isce.geocode.GeocodeOutputMode.AREA_PROJECTION

        # only 2 RTC algorithms supported: area_projection (default) & bilinear_distribution
        if rtc_dict['algorithm_type'] == "bilinear_distribution":
            rtc_dict['algorithm_type'] = isce.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION
        else:
            rtc_dict['algorithm_type'] = isce.geometry.RtcAlgorithm.RTC_AREA_PROJECTION

        if rtc_dict['input_terrain_radiometry'] == "sigma0":
            rtc_dict['input_terrain_radiometry'] = isce.geometry.RtcInputTerrainRadiometry.SIGMA_NAUGHT_ELLIPSOID
        else:
            rtc_dict['input_terrain_radiometry'] = isce.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT

        if rtc_dict['rtc_min_value_db'] is None:
            rtc_dict['rtc_min_value_db'] = np.nan

        if self.cfg['processing']['dem_margin'] is None:
            '''
            Default margin as the length of 50 pixels
            (max of X and Y pixel spacing).
            '''
            dem_file = self.cfg['DynamicAncillaryFileGroup']['DEMFile']
            dem_raster = isce.io.Raster(dem_file)
            dem_margin = 50 * max([dem_raster.dx, dem_raster.dy])
            self.cfg['processing']['dem_margin'] = dem_margin
