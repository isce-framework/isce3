import journal

import pybind_isce3 as isce
from pybind_nisar.workflows.geo2rdr_runconfig import Geo2rdrRunConfig

class InsarRunConfig(Geo2rdrRunConfig):
    def __init__(self, args):
        super().__init__(args)
        self.load_geocode_yaml_to_dict()
        self.geocode_common_arg_load()
        self.yaml_check()

    def yaml_check(self):
        '''
        Check submodule paths from YAML
        '''
        scratch_path = self.cfg['ProductPathGroup']['ScratchPath']

        # for each submodule check if user path for input data assigned
        # if not assigned, assume it'll be in scratch
        if 'topo_path' not in self.cfg['processing']['geo2rdr']:
            self.cfg['processing']['geo2rdr']['topo_path'] = scratch_path

        if 'offset_dir' not in self.cfg['processing']['resample']:
            self.cfg['processing']['resample']['offset_dir'] = scratch_path

        if 'coregistered_slc_path' not in self.cfg['processing']['crossmul']:
            self.cfg['processing']['crossmul']['coregistered_slc_path'] = scratch_path

        flatten = self.cfg['processing']['crossmul']['flatten']
        if flatten:
            if isinstance(flatten, bool):
                self.cfg['processing']['crossmul']['flatten'] = scratch_path
        else:
            self.cfg['processing']['crossmul']['flatten'] = None

        # set to empty dict and default unwrap values will be used
        # if phase_unwrap fields not in yaml
        if 'phase_unwrap' not in self.cfg['processing']:
           self.cfg['processing']['phase_unwrap'] = {}

        # if phase_unwrap fields not in yaml
        if self.cfg['processing']['phase_unwrap'] is None:
           self.cfg['processing']['phase_unwrap'] = {}
