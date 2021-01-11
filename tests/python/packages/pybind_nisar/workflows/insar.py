import argparse
import os

import numpy as np

from pybind_nisar.workflows import h5_prep, insar
from pybind_nisar.workflows.insar_runconfig import InsarRunConfig
from pybind_nisar.workflows.persistence import Persistence

import iscetest

def test_insar_run():
    '''
    Run InSAR run
    '''

    # load yaml file
    test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')

    # Load text and substitute directory path
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data).\
                replace('@TEST_OUTPUT@', 'insar.h5').\
                replace('@TEST_PRODUCT_TYPES@', 'GUNW')

    # create CLI input namespace
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # Initialize runconfig object
    insar_runcfg = InsarRunConfig(args)
    insar_runcfg.geocode_common_arg_load()
    insar_runcfg.yaml_check()

    out_paths = h5_prep.run(insar_runcfg.cfg)

    persist = Persistence(restart=True)

    insar.run(insar_runcfg.cfg, out_paths, persist.run_steps)

if __name__ == "__main__":
    test_insar_run()
