import numpy as np

import journal

import isce3
from nisar.workflows.runconfig import RunConfig

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
            geocode_dict['clip_max'] = np.nan

        if geocode_dict['clip_min'] is None:
            geocode_dict['clip_min'] = np.nan

        if geocode_dict['geogrid_upsampling'] is None:
            geocode_dict['geogrid_upsampling'] = 1.0

        if geocode_dict['memory_mode'] == 'single_block':
            geocode_dict['memory_mode'] = isce3.geocode.GeocodeMemoryMode.SINGLE_BLOCK
        elif geocode_dict['memory_mode'] == 'geogrid':
            geocode_dict['memory_mode'] = isce3.geocode.GeocodeMemoryMode.BLOCKS_GEOGRID
        elif geocode_dict['memory_mode'] == 'geogrid_and_radargrid':
            geocode_dict['memory_mode'] = isce3.geocode.GeocodeMemoryMode.BLOCKS_GEOGRID_AND_RADARGRID
        elif geocode_dict['memory_mode'] == 'auto':
            geocode_dict['memory_mode'] = isce3.geocode.GeocodeMemoryMode.AUTO
        else:
            err_msg = f"ERROR memory_mode: {geocode_dict['memory_mode']}"
            raise ValueError(err_msg)

        rtc_output_type = rtc_dict['output_type']
        if rtc_output_type == 'sigma0':
            rtc_dict['output_type'] = isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT
        else:
            rtc_dict['output_type'] = isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT


        geocode_algorithm = self.cfg['processing']['geocode']['algorithm_type']
        if geocode_algorithm == "area_projection":
            output_mode = isce3.geocode.GeocodeOutputMode.AREA_PROJECTION
        else:
            output_mode = isce3.geocode.GeocodeOutputMode.INTERP
        geocode_dict['output_mode'] = output_mode

        # only 2 RTC algorithms supported: area_projection (default) & bilinear_distribution
        if rtc_dict['algorithm_type'] == "bilinear_distribution":
            rtc_dict['algorithm_type'] = isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION
        else:
            rtc_dict['algorithm_type'] = isce3.geometry.RtcAlgorithm.RTC_AREA_PROJECTION

        if rtc_dict['input_terrain_radiometry'] == "sigma0":
            rtc_dict['input_terrain_radiometry'] = isce3.geometry.RtcInputTerrainRadiometry.SIGMA_NAUGHT_ELLIPSOID
        else:
            rtc_dict['input_terrain_radiometry'] = isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT

        if rtc_dict['rtc_min_value_db'] is None:
            rtc_dict['rtc_min_value_db'] = np.nan

        if self.cfg['processing']['dem_margin'] is None:
            '''
            Default margin as the length of 50 pixels
            (max of X and Y pixel spacing).
            '''
            dem_file = self.cfg['dynamic_ancillary_file_group']['dem_file']
            dem_raster = isce3.io.Raster(dem_file)
            dem_margin = 50 * max([dem_raster.dx, dem_raster.dy])
            self.cfg['processing']['dem_margin'] = dem_margin

        # Update the DEM interpolation method
        dem_interp_method = \
            self.cfg['processing']['dem_interpolation_method']

        if dem_interp_method == 'biquintic':
            dem_interp_method_enum = isce3.core.DataInterpMethod.BIQUINTIC
        elif (dem_interp_method == 'sinc'):
            dem_interp_method_enum = isce3.core.DataInterpMethod.SINC
        elif (dem_interp_method == 'bilinear'):
            dem_interp_method_enum = isce3.core.DataInterpMethod.BILINEAR
        elif (dem_interp_method == 'bicubic'):
            dem_interp_method_enum = isce3.core.DataInterpMethod.BICUBIC
        elif (dem_interp_method == 'nearest'):
            dem_interp_method_enum = isce3.core.DataInterpMethod.NEAREST
        else:
            err_msg = ('ERROR invalid DEM interpolation method:'
                       f' {dem_interp_method}')
            raise ValueError(err_msg)

        self.cfg['processing']['dem_interpolation_method_enum'] = \
            dem_interp_method_enum
