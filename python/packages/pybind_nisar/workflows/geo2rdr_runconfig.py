import pybind_nisar.workflows.helpers as helpers
from pybind_nisar.workflows.runconfig import RunConfig


class Geo2rdrRunConfig(RunConfig):
    def __init__(self, args):
        # InSAR submodules share a common "InSAR" schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            self.load_geocode_yaml_to_dict()
            self.geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        '''
        Check geo2rdr specifics from YAML.
        '''
        # Use scratch as topo_path if none given in YAML
        if 'topo_path' not in self.cfg['processing']['geo2rdr']:
            self.cfg['processing']['geo2rdr']['topo_path'] = self.cfg['ProductPathGroup']['ScratchPath']

        # Check topo directory structure
        topo_path = self.cfg['processing']['geo2rdr']['topo_path']
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        frequencies = freq_pols.keys()
        helpers.check_mode_directory_tree(topo_path, 'rdr2geo', frequencies)
