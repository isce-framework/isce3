import os

import journal
from nisar.workflows.runconfig import RunConfig


class SplitMainBandRunConfig(RunConfig):
    def __init__(self, args):
        # all insar submodules share a commmon `insar` schema
        super().__init__(args, 'insar')

        self.load_geocode_yaml_to_dict()
        self.geocode_common_arg_load()
        self.yaml_check()

    def yaml_check(self):
        '''
        Check split_main_band specifics from YAML
        '''
        error_channel = journal.error('SplitMainBandRunConfig.yaml_check')

        window_type = self.cfg['processing']['ionosphere_correction']['window_function']
        method = self.cfg['processing']['ionosphere_correction']['method']

        if window_type.lower() not in ['kaiser', 'cosine', 'tukey']:
            err_str = f"{window_type} not a valid window type"
            error_channel.log(err_str)
            raise ValueError(err_str)

        if method.lower() not in ['split_main_band', 'main_side']:
            err_str = f"{method} not a valid method type"
            error_channel.log(err_str)
            raise NotImplementedError(err_str)