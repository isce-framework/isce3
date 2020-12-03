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
        if 'topo_path' not in self.cfg['processing']['geo2rdr']:
            self.cfg['processing']['geo2rdr']['topo_path'] = self.cfg['ProductPathGroup']['ScratchPath']
        if 'offset_dir' not in self.cfg['processing']['resample']:
            self.cfg['processing']['resample']['offset_dir'] = self.cfg['ProductPathGroup']['ScratchPath']
