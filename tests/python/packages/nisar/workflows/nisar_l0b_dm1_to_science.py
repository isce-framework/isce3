import os
import argparse

import pytest

from nisar.workflows.nisar_l0b_dm1_to_science import nisar_l0b_dm1_to_science
import iscetest


@pytest.mark.parametrize(
        "prod_name,ovsf_rg,plot,num_cpu",
        [
            (None, None, False, None),
            ('dm1.h5', 1.2, True, 4)
        ],
        ids=["default", "non-default"]
)
def test_nisar_l0b_dm1_to_science(prod_name, ovsf_rg, plot, num_cpu):
    # sub directory for all test files under "isce3/tests/data"
    sub_dir = 'dm1_dm2'
    # Simulated single-pol single-band NISAR-like DM1 L0B product
    l0b_file = 'REE_L0B_ECHO_DM1_240MHZ_RX1.h5'

    # input path for all input files
    p_in = os.path.join(iscetest.data, sub_dir)

    # form args
    args = argparse.Namespace(
        l0b_file=os.path.join(p_in, l0b_file),
        out_path='.',
        prod_name=prod_name,
        num_cpu=num_cpu,
        num_rgl=5000,
        comp_level_h5=4,
        plot=plot,
        ovsf_rg=ovsf_rg,
        sign_mag=False,
        nbits=12
    )
    nisar_l0b_dm1_to_science(args)
