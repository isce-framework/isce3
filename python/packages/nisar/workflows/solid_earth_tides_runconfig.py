#!/usr/bin/env python3
from nisar.workflows.runconfig import RunConfig

class InsarSolidEarthTidesRunConfig(RunConfig):
    '''
    Solid Earth Tides RunConfig
    '''

    def __init__(self, args):
        # InSAR submodules share a common "InSAR" schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
