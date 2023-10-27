import journal
from nisar.workflows.runconfig import RunConfig


class BaselineRunConfig(RunConfig):
    def __init__(self, args):
        # all insar submodules share a commmon `insar` schema
        super().__init__(args, 'insar')

        self.load_geocode_yaml_to_dict()
        self.geocode_common_arg_load()
        self.yaml_check()

    def yaml_check(self):
        '''
        Check baseline specifics from YAML
        '''
        error_channel = journal.error('BaselineRunConfig.yaml_check')

        mode_type = self.cfg['processing']['baseline']['mode']

        if mode_type.lower() not in ['3d_full', 'top_bottom']:
            err_str = f"{mode_type} not a valid baseline estimation mode"
            error_channel.log(err_str)
            raise ValueError(err_str)

