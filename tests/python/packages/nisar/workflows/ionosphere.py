import argparse
import os

import iscetest
from nisar.workflows import insar
from nisar.workflows.h5_prep import get_products_and_paths
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.persistence import Persistence


def test_split_main_band_run():
    '''
    Check if split_main_band runs without crashing
    '''

    # Load yaml file
    test_yaml = os.path.join(iscetest.data, 'ionosphere_test.yaml')
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data). \
            replace('@TEST_OUTPUT@', 'RUNW.h5'). \
            replace('@TEST_PRODUCT_TYPES@', 'RUNW'). \
            replace('@TEST_RDR2GEO_FLAGS@', 'True'). \
            replace('spectral_diversity:', 'spectral_diversity: split_main_band')

    # Create CLI input namespace with yaml text instead of filepath
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # Initialize runconfig object
    insar_runcfg = InsarRunConfig(args)
    insar_runcfg.geocode_common_arg_load()
    insar_runcfg.yaml_check()

    _, out_paths = get_products_and_paths(insar_runcfg.cfg)
    persist = Persistence(restart=True, logfile_path='ionosphere.log')

    # No CPU dense offsets. Turn off dense_offsets,
    # rubbersheet, and fine_resample to avoid test failure
    persist.run_steps['dense_offsets'] = False
    persist.run_steps['rubbersheet'] = False
    persist.run_steps['fine_resample'] = False
    persist.run_steps['baseline'] = False

    # run insar for prod_type
    insar.run(insar_runcfg.cfg, out_paths, persist.run_steps)


def test_main_side_band_run():
    '''
    Check if main_side_band runs without crashing
    '''

    # Load yaml file
    test_yaml = os.path.join(iscetest.data, 'ionosphere_main_side_test.yaml')
    with open(test_yaml) as fh_test_yaml:
        test_yaml = fh_test_yaml.read().replace('@ISCETEST@', iscetest.data). \
            replace('@TEST_OUTPUT@', 'RUNW.h5'). \
            replace('@TEST_PRODUCT_TYPES@', 'RUNW'). \
            replace('@TEST_RDR2GEO_FLAGS@', 'True'). \
            replace('spectral_diversity:', 'spectral_diversity: main_side_band')

    # Create CLI input namespace with yaml text instead of filepath
    args = argparse.Namespace(run_config_path=test_yaml, log_file=False)

    # Initialize runconfig object
    insar_runcfg = InsarRunConfig(args)
    insar_runcfg.geocode_common_arg_load()
    insar_runcfg.yaml_check()

    _, out_paths = get_products_and_paths(insar_runcfg.cfg)

    persist = Persistence(restart=True, logfile_path='ionosphere.log')
    # No CPU dense offsets. Turn off dense_offsets,
    # rubbersheet, and fine_resample to avoid test failure
    persist.run_steps['dense_offsets'] = False
    persist.run_steps['rubbersheet'] = False
    persist.run_steps['fine_resample'] = False
    # the baseline step is disabled because the winnipeg test dataset
    # is missing some required metadata.
    persist.run_steps['baseline'] = False

    # run insar for prod_type
    insar.run(insar_runcfg.cfg, out_paths, persist.run_steps)


if __name__ == '__main__':
    test_split_main_band_run()
    test_main_side_band_run()
