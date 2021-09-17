from collections import defaultdict
import os

import journal

from nisar.workflows.runconfig import RunConfig
import nisar.workflows.helpers as helpers


class BandpassRunConfig(RunConfig):
    def __init__(self, args):
        # all insar submodules share a commmon `insar` schema
        super().__init__(args, 'insar')

        self.load_geocode_yaml_to_dict()
        self.geocode_common_arg_load()
        self.yaml_check()