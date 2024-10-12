import os
import argparse

import pytest

from nisar.workflows.nisar_l0b_dm2_to_dbf import nisar_l0b_dm2_to_dbf
import iscetest


@pytest.mark.parametrize(
        "no_rgcomp,calib,plot,prod_name",
        [
            (True, False, False, "dm2_seamed.h5"),
            (False, True, True, None)
        ],
        ids=["seamed", "seamless"]
)
def test_nisar_l0b_dm2_to_dbf(no_rgcomp, calib, plot, prod_name):
    # sub directory for all test files under "isce3/tests/data"
    sub_dir = 'dm1_dm2'
    # Simulated single-pol single-band NISAR-like DM2 L0B product
    # plus antenna pattern, orbit, attitude, and DEM files
    l0b_file = 'REE_L0B_ECHO_DM2.h5'
    ant_file = 'REE_ANTPAT_CUTS_DM2.h5'
    orbit_file = 'NISAR_ANC_L_PR_FOE_20240912T151522_20240119T221330_20240119T221340.xml'
    attitude_file = 'NISAR_ANC_L_PR_FRP_20240912T151522_20240119T221330_20240119T221340.xml'
    dem_file = 'FIXED_HEIGHT_DM2.tif'

    # input path for all input files
    p_in = os.path.join(iscetest.data, sub_dir)

    # form args
    args = argparse.Namespace(
        l0b_file=os.path.join(p_in, l0b_file),
        ant_file=os.path.join(p_in, ant_file),
        orbit_file=os.path.join(p_in, orbit_file),
        attitude_file=os.path.join(p_in, attitude_file),
        dem_file=os.path.join(p_in, dem_file),
        ref_height=0.0,
        out_path='.',
        num_cpu=None,
        no_rgcomp=no_rgcomp,
        num_rgl=5000,
        comp_level_h5=4,
        plot=plot,
        multiplier=None,
        win_ped=1.0,
        calib=calib,
        amp_cal=None,
        prod_name=prod_name
    )
    nisar_l0b_dm2_to_dbf(args)
