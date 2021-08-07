import argparse
import os

import pybind_isce3 as isce3
from nisar.workflows import h5_prep, insar
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.persistence import Persistence

import iscetest


def test_insar_run():
    '''
    Run InSAR run
    '''

    # load yaml file
    path_test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')

    for prod_type in ['RIFG', 'RUNW', 'GUNW']:
        test_output = f'{prod_type}.h5'

        # Load text and substitute directory path
        with open(path_test_yaml) as fh_test_yaml:
            test_yaml = fh_test_yaml.read().replace('@ISCETEST@',
                                                    iscetest.data). \
                replace('@TEST_OUTPUT@', test_output). \
                replace('@TEST_PRODUCT_TYPES@', prod_type). \
                replace('gpu_enabled: False', 'gpu_enabled: True')

        # create CLI input namespace
        args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

        # Initialize runconfig object
        insar_runcfg = InsarRunConfig(args)
        insar_runcfg.geocode_common_arg_load()
        insar_runcfg.yaml_check()

        out_paths = h5_prep.run(insar_runcfg.cfg)

        persist = Persistence(restart=True)

        # run insar for prod_type
        insar.run(insar_runcfg.cfg, out_paths, persist.run_steps)

        # check if test_output exists
        assert os.path.isfile(test_output), f"{test_output} for {prod_type} not found."


if __name__ == "__main__":
    test_insar_run()
