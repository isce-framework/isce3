import journal

import isce3
from nisar.workflows.runconfig import RunConfig

class GSLCRunConfig(RunConfig):
    def __init__(self, args):
        super().__init__(args, 'gslc')
        super().load_geocode_yaml_to_dict()
        super().geocode_common_arg_load()
