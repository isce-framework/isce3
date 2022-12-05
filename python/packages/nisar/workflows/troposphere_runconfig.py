import journal
import os
import h5py
import numpy as np

from nisar.workflows.geo2rdr_runconfig import Geo2rdrRunConfig

class InsarTroposphereRunConfig(Geo2rdrRunConfig):
    '''
    Troposhphere RunConfig
    '''

    def __init__(self, args):
        super().__init__(args)
        self.load_geocode_yaml_to_dict()
        self.geocode_common_arg_load()
        self.yaml_check()

    def yaml_check(self):
        '''
        Check submodule paths from YAML
        '''
        
        error_channel = journal.error('InsarTroposphereRunConfig.yaml_check')
        info_channel = journal.applicationfo('InsarTroposphereRunConfig.yaml_check')
        
        # Check the weater model files

        dynamic_weather_model_cfg = self.cfg['dynamic_ancillary_file_group']['weather_model']
        
        weather_model = dynamic_weather_model_cfg['weather_model']
        if weather_model.upper() not in ['ERA5', 'ERAINT', 'HRES', 'NARR', 'MERRA']:
            weather_model = 'ERA5'

        ref_weather_model_file = dynamic_weather_model_cfg['reference_weather_model_file_path']
        sec_weather_model_file = dynamic_weather_model_cfg['secondary_weather_model_file_path']

        if ref_weather_model_file is None:
            err_str = 'reference weather model file cannot be None, please specify the weather model'
            error_channel.log(err_str)
            raise ValueError(err_str)

        if sec_weather_model_file is None:
            err_str = 'secondary weather model file cannot be None, please specify the weather model'
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Create defaults for troposphere delay computation
        tropo_cfg = self.cfg['processing']['troposphere_delay']
        
        # not sure if we need this, but put it here just in case
        #cfg['primary_executable']['product_type'] = "GUNW"

        # The default is not enabled
        if tropo_cfg['enabled'] is None:
            tropo_cfg['enabled'] = False
        
        # Check the external package, if no package is assigned, the 'pyaps' will be applied 
        if tropo_cfg['package'].lower() not in ['pyaps','raider']:
            info_channel.log(f"unidentified package '{tropo_cfg['package']}', the 'pyaps' package will be applied")
            tropo_cfg['package'] = 'payaps'

        # Check the delay direction
        if tropo_cfg['delay_direction'].lower() not in ['zenith','line_of_sight_mapping','line_of_sight_raytracing']:
            info_channel.log(f"unidentified delay direction '{tropo_cfg['delay_direction']}', the 'line_of_sight delay will be applied'")
            tropo_cfg['delay_direction'] = 'line_of_sight_mapping'
        
        # If the delay direction is 'line_of_sight_raytracing', the package must be 'raider'
        if tropo_cfg['delay_direction'].lower() == 'line_of_sight_raytracing' and tropo_cfg['package'] != 'raider':
            err_str = "for line_of_sight_raytracing delay type, the package must be 'raider'"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # Check the troposphere delay product
        if not isinstance(tropo_cfg['delay_product'],list):
            err_str = "the inputs of the delay_product should be the list type (e.g. ['combo'])"
            raise ValueError(err_str)
        else:
            for delay_product in tropo_cfg['delay_product']:
                if delay_product.lower() not in ['wet', 'hydro', 'comb']:
                    err_str = f"unidentified delay product '{tropo_cfg['delay_product']}', it should be one or more of 'wet', 'hydro', and 'comb'"
                    raise ValueError(err_str)

            # Check if it is an empty list
            if len(tropo_cfg['delay_product']) == 0:
                info_channel.log("the delay product is empty, the 'comb' will be applied")
                tropo_cfg['delay_product'] = ['comb']
