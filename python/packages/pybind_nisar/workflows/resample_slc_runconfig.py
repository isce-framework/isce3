import pybind_nisar.workflows.helpers as helpers
from pybind_nisar.workflows.runconfig import RunConfig


class ResampleSlcRunConfig(RunConfig):
    def __init__(self, args):
        # InSAR submodules have a common InSAR schema
        super().__init__(args, 'insar')

        if self.args.run_config_path is not None:
            super().load_geocode_yaml_to_dict()
            super().geocode_common_arg_load()
            self.yaml_check()

    def yaml_check(self):
        '''
        Check resample specifics from YAML.
        '''
        # Use scratch as offset_dir if none given in YAML
        if 'offset_dir' not in self.cfg['processing']['resample']:
            self.cfg['processing']['resample']['offset_dir'] = self.cfg['ProductPathGroup']['ScratchPath']

        # Check offsets directory structure
        off_dir = self.cfg['processing']['resample']['offset_dir']
        freq_pols = self.cfg['processing']['input_subset']['list_of_frequencies']
        frequencies = freq_pols.keys()
        helpers.check_mode_directory_tree(off_dir, 'geo2rdr', frequencies)
