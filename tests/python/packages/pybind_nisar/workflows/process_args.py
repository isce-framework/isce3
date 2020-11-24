import subprocess

import numpy.testing as npt

import pybind_nisar.workflows.yaml_argparse as yaml_argparse
import pybind_nisar.workflows.rdr2geo_argparse as rdr2geo_argparse

import iscetest


def test_cli_interface():
    '''
    run YAML and CLI success and failures
    '''
    # test YAML run success. iscetest.data write only so log file off.
    process_args_path = yaml_argparse.__file__
    cmd = f'python {process_args_path} {iscetest.data}/insar_test.yaml'
    proc = subprocess.run(cmd.split())

    # test YAML run fail with bad path
    cmd = f'python {process_args_path} {iscetest.data}/no_here.yaml'
    with npt.assert_raises(subprocess.CalledProcessError):
        proc = subprocess.check_output(cmd.split())

    # test simulated CLI success
    process_args_path = rdr2geo_argparse.__file__
    cmd = f'python {process_args_path} --rdr2geo \
        --input-h5 {iscetest.data}/envisat.h5 \
        --dem {iscetest.data}/srtm_cropped.tif \
        --scratch . \
        --frequencies-polarizations A=HH,HV B=VV'
    proc = subprocess.run(cmd.split())


if __name__ == "__main__":
    test_cli_interface()
