#!/usr/bin/env python3

import journal

from nisar.workflows.runconfig import RunConfig


def solidearth_tides_check(cfg):
    '''
    Check the solid earth tides yaml

    Parameters
     ----------
     cfg: dict
        configuration dictionary

     Returns
     -------
     None
    '''

    error_channel = journal.error(
        'InsarSolidEarthTidesRunConfig.solidearth_tides_check')
    info_channel = journal.info(
        'InsarSolidEarthTidesRunConfig.solidearth_tides_check')

    solidearth_tides_cfg = cfg['processing']['solidearth_tides']

    # Only if the solid earth tides  is enabled
    if solidearth_tides_cfg['enabled']:

        step_size = solidearth_tides_cfg['step_size']

        if not isinstance(step_size, (int, float)):
            err_str = 'the data type of the step size should be int or float'
            error_channel.log(err_str)
            raise ValueError(err_str)


class InsarSolidEarthTidesRunConfig(RunConfig):
    '''
    Solid Earth Tides  RunConfig
    '''

    def __init__(self, args):
        # InSAR submodules share a common "InSAR" schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            solidearth_tides_check(self.cfg)
