import journal
from nisar.workflows.runconfig import RunConfig


class BandpassRunConfig(RunConfig):
    def __init__(self, args):
        # all insar submodules share a common `insar` schema
        super().__init__(args, 'insar')

        self.load_geocode_yaml_to_dict()
        self.geocode_common_arg_load()
        self.yaml_check()

    def yaml_check(self):
        '''
        Check bandpass specifics from YAML
        '''
        error_channel = journal.error('BandpassRunConfig.yaml_check')

        window_type = self.cfg['processing']['bandpass']['window_function']

        if window_type.lower() not in ['kaiser', 'cosine', 'tukey']:
            err_str = f"{window_type} not a valid window type"
            error_channel.log(err_str)
            raise ValueError(err_str)