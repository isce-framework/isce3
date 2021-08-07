import subprocess

import numpy.testing as npt

import iscetest


def test_cli_interface():
    '''
    run YAML and CLI success and failures
    '''
    # test YAML run success. iscetest.data write only so log file off.
    cmd = f'python3 -m nisar.workflows.yaml_argparse \
            {iscetest.data}/insar_test.yaml --no-log-file'
    proc = subprocess.run(cmd.split())

    # test YAML run fail with bad path
    cmd = f'python3 -m nisar.workflows.yaml_argparse \
            {iscetest.data}/no_here.yaml --no-log-file'
    with npt.assert_raises(subprocess.CalledProcessError):
        proc = subprocess.check_output(cmd.split())

    # test simulated CLI success
    cmd = f'python3 -m nisar.workflows.rdr2geo_argparse \
        --input-h5 {iscetest.data}/envisat.h5 \
        --dem {iscetest.data}/srtm_cropped.tif \
        --scratch . \
        --frequencies-polarizations A=HH,HV B=VV'
    proc = subprocess.run(cmd.split())


if __name__ == "__main__":
    test_cli_interface()
