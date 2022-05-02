import argparse
import os
from osgeo import gdal

import h5py
import iscetest
import numpy as np
from isce3.ionosphere import ionosphere_estimation
from nisar.workflows import h5_prep, insar
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.persistence import Persistence


def simulate_ifgrams(f, dr, dTEC):
    speed_of_light = 299792458.0
    phi_non_dispersive = 4.0*np.pi*f*dr/speed_of_light
    K = 40.31
    phi_TEC = (-4.0*np.pi*K/(speed_of_light*f))*dTEC
    phi = phi_non_dispersive + phi_TEC
    
    return phi, phi_non_dispersive , phi_TEC#

def test_ionosphere_methods():
    f0 = 1.27e9
    f1 = f0 + 60.0e6

    BW = 20.0e6 # bandwidth of the main band

    f0L = f0 - BW/3.0
    f0H = f0 + BW/3.0
    dr = np.array([[0.2]])
    dTEC = np.array([[2.0*1e16]])
    phi0, phi0_non, phi0_iono = simulate_ifgrams(f0, dr, dTEC)
    phi1, j0, j1 = simulate_ifgrams(f1, dr, dTEC)
    phi0L, j0, j1 = simulate_ifgrams(f0L, dr, dTEC)
    phi0H, j0, j1 = simulate_ifgrams(f0H, dr, dTEC)

    phi0_LH = phi0H - phi0L
    phi_ms = phi0 - phi1    

    iono_obj = ionosphere_estimation.IonosphereEstimation(
        main_center_freq=f0,
        side_center_freq=f1, 
        low_center_freq=f0L, 
        high_center_freq=f0H,
        method='split_main_band')
        
    phi_n_LH, phi_iono_LH = iono_obj.estimate_iono_low_high(
            f0=f0, 
            freq_low=f0L, 
            freq_high=f0H, 
            phi0_low=phi0L, 
            phi0_high=phi0H)
    
    phi_n_ms, phi_iono_ms = iono_obj.estimate_iono_main_side(
            f0=f0, 
            f1=f1, 
            phi0=phi0, 
            phi1=phi1)
    
    phi_n_md, phi_iono_md = iono_obj.estimate_iono_main_diff(
            f0=f0, 
            f1=f1, 
            phi0=phi0, 
            phi1=phi1)

    difference_lh_ms_abs = np.abs(phi_n_LH - phi_n_ms)
    difference_lh_md_abs = np.abs(phi_n_LH - phi_n_md)

    assert difference_lh_ms_abs < 1e-5
    assert difference_lh_md_abs < 1e-5

def test_split_main_band_run():
    '''
    Check if split_main_band runs without crashing
    '''

    # Load yaml file
    test_yaml = os.path.join(iscetest.data, 'insar_test.yaml')
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

    out_paths = h5_prep.run(insar_runcfg.cfg)
    persist = Persistence(restart=True)

    # No CPU dense offsets. Turn off dense_offsets,
    # rubbersheet, and fine_resample to avoid test failure
    persist.run_steps['dense_offsets'] = False
    persist.run_steps['rubbersheet'] = False
    persist.run_steps['fine_resample'] = False

    # run insar for prod_type
    insar.run(insar_runcfg.cfg, out_paths, persist.run_steps)

if __name__ == '__main__':
    test_ionosphere_methods()
    test_split_main_band_run()
