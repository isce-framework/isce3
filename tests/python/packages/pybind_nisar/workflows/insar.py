import argparse
import os

import numpy as np

from pybind_nisar.workflows import insar
from pybind_nisar.workflows.insar_runconfig import InsarRunConfig

import iscetest


def test_insar_run():
    '''
    Run InSAR run
    '''

    # load yaml file
    test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')

    # Load text and substitute directory path
    with open(test_yaml) as fh_test_yaml:
        test_yaml = ''.join(
            [line.replace('ISCETEST', iscetest.data) for line in fh_test_yaml])

    # create CLI input namespace
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # Initialize runconfig object
    insar_runcfg = InsarRunConfig(args)
    insar_runcfg.geocode_common_arg_load()
    insar.run(insar_runcfg.cfg)
