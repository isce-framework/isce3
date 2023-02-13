#!/usr/bin/env python3

import journal

from nisar.workflows.runconfig import RunConfig


def solid_earth_tides_check(cfg):
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

    solid_earth_tides_cfg = cfg['processing']['solid_earth_tides']

    # Only if the solid earth tides  is enabled
    if solid_earth_tides_cfg['enabled']:
        pass


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
