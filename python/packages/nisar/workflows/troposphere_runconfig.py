#!/usr/bin/env python3

import h5py
import journal
import numpy as np
import os

from nisar.workflows.geo2rdr_runconfig import Geo2rdrRunConfig

def troposphere_delay_check(cfg):
    '''
    Check the troposphere delay yaml

    Parameters
     ----------
     cfg: dict
        configuration dictionary

     Returns
     -------
     None
    '''

    error_channel = journal.error('InsarTroposphereRunConfig.yaml_check')
    info_channel = journal.info('InsarTroposphereRunConfig.yaml_check')
    
    tropo_cfg = cfg['processing']['troposphere_delay']

    # only if the troposphere is enabled
    if tropo_cfg['enabled']:

        # Check the weater model files
        dynamic_weather_model_cfg = cfg['dynamic_ancillary_file_group']['troposphere_weather_model']
    
        for option in ['reference','secondary']:

            weather_model_file = dynamic_weather_model_cfg[f'{option}_file_path']

            if (weather_model_file is None) or (not os.path.exists(weather_model_file)):
                err_str = f'{option} weather model file cannot be None or not found,\
                        please specify the {option} weather model'
                error_channel.log(err_str)
                raise ValueError(err_str)


        weather_model_type = tropo_cfg['weather_model_type'].upper()
        
        # Check the weather model
        if weather_model_type not in ['ERA5', 'ERAINT', 'HRES', 'NARR', 'MERRA',
                                      'ECWMF', 'ERAI', 'GMAO', 'HRRR', 'NCMR']:

            err_str = f"unidentified weather model, {weather_model_type},\
                        please try one of \
                        'ERA5', 'ERAINT', 'HRES', 'NARR', 'MERRA', 'ECWMF', 'ERAI', 'GMAO', 'HRRR', 'NCMR'"

            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check the external tropo delay  package
        if tropo_cfg['package'].lower() not in ['pyaps', 'raider']:
            err_str = f"unidentified package {tropo_cfg['package']}, please use either pyaps or raider"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Further check the weather model that supported by pyAPS or RAiDER
        if (tropo_cfg['package'].lower() == 'pyaps') and \
                weather_model_type not in ['ERA5', 'ERAINT', 'HRES', 'NARR', 'MERRA']:
            err_str = f'weather model {weather_model_type} is not supported by pyAPS package'
            error_channel.log(err_str)
            raise ValueError(err_str)

        if (tropo_cfg['package'].lower() == 'raider') and \
                weather_model_type not in ['ERA5', 'HRES', 'MERRA', 'ECWMF', 'ERAI', 'GMAO', 'HRRR', 'NCMR']:
            err_str = f'weather model {weather_model_type} is not supported by RAiDER  package'
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check the delay direction
        if tropo_cfg['delay_direction'].lower() not in ['zenith', 'line_of_sight_mapping', 'line_of_sight_raytracing']:
            err_str = f"unidentified delay direction {tropo_cfg['delay_direction']},\
                        please use one of 'zenith', 'line_of_sight_mapping', 'line_of_sight_raytracing'"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # If the delay direction is 'line_of_sight_raytracing', the package must be 'raider'
        if tropo_cfg['delay_direction'].lower() == 'line_of_sight_raytracing' and tropo_cfg['package'].lower() != 'raider':
            err_str = "for line_of_sight_raytracing delay type, the package must be 'raider'"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check the troposphere delay product
        if not isinstance(tropo_cfg['delay_product'], list):
            err_str = "the inputs of the delay_product should be the list type (e.g. ['comb'])"
            error_channel.log(err_str)
            raise ValueError(err_str)
        else:
            for delay_product in tropo_cfg['delay_product']:
                if delay_product.lower() not in ['wet', 'hydro', 'comb']:
                    err_str = f"unidentified delay product '{tropo_cfg['delay_product']}', \
                                it should be one or more of 'wet', 'hydro', and 'comb'"
                    error_channel.log(err_str)
                    raise ValueError(err_str)

            # Check if it is an empty list
            if len(tropo_cfg['delay_product']) == 0:
                info_channel.log(
                    "the delay product is empty, the 'comb' will be applied")
                tropo_cfg['delay_product'] = ['comb']


class InsarTroposphereRunConfig(Geo2rdrRunConfig):
    ''' 
    Troposhphere RunConfig
    '''

    def __init__(self, args):
        super().__init__(args)
        self.load_geocode_yaml_to_dict()
        self.geocode_common_arg_load()
        troposphere_delay_check(self.cfg)
