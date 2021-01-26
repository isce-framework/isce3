import iscetest
import numpy as np
import numpy.testing as npt
from pybind_nisar.workflows import point_target_info as pt

def test_islr_pslr():
    num_pts = 2000
    theta = np.linspace(-15, 15, num_pts)
    ant_pat = np.sinc(theta)
   
    #Pass/Fail
    pslr_ideal_dB = -13.26
    islr_ideal_single_dB = -10
    islr_ideal_double_dB = -13.5
    pslr_max_err = 0.1
    islr_max_err = 0.25

    #Test Case 1: num of nulls in the mainlobe = 1, search for nulls
    fs_bw_ratio = 68
    num_nulls_main = 1
    num_lobes = 11
    search_null = True

    islr_single_dB, pslr_dB = pt.islr_pslr(ant_pat, fs_bw_ratio=fs_bw_ratio, num_nulls_main=num_nulls_main, num_lobes=num_lobes, search_null=search_null)
    pslr_err = np.abs(pslr_dB - pslr_ideal_dB)
    islr_err = np.abs(islr_single_dB - islr_ideal_single_dB)

    npt.assert_array_less(pslr_err, pslr_max_err, 'PSLR error is larger than expected')
    npt.assert_array_less(islr_err, islr_max_err, 'ISLR error is larger than expected')

    #Test Case 2: num of nulls in the mainlobe = 1, Use default fs/bw ratio to locate nulls
    search_null = False
    islr_single_dB, pslr_dB = pt.islr_pslr(ant_pat, fs_bw_ratio=fs_bw_ratio, num_nulls_main=num_nulls_main, num_lobes=num_lobes, search_null=search_null)
    pslr_err = np.abs(pslr_dB - pslr_ideal_dB)
    islr_err = np.abs(islr_single_dB - islr_ideal_single_dB)

    npt.assert_array_less(pslr_err, pslr_max_err, 'PSLR error is larger than expected')
    npt.assert_array_less(islr_err, islr_max_err, 'ISLR error is larger than expected')

    #Test Case 3: num of nulls in the mainlobe = 2, search for nulls
    fs_bw_ratio = 68
    num_nulls_main = 2
    num_lobes = 12
    search_null = True

    islr_double_dB, pslr_dB = pt.islr_pslr(ant_pat, fs_bw_ratio=fs_bw_ratio, num_nulls_main=num_nulls_main, num_lobes=num_lobes, search_null=search_null)
    pslr_err = np.abs(pslr_dB - pslr_ideal_dB)
    islr_err = np.abs(islr_double_dB - islr_ideal_double_dB)

    npt.assert_array_less(pslr_err, pslr_max_err, 'PSLR error is larger than expected')
    npt.assert_array_less(islr_err, islr_max_err, 'ISLR error is larger than expected')

    #Test Case 4: num of nulls in the mainlobe = 2, use default fs/bw ratio to locate nulls
    search_null = False

    islr_double_dB, pslr_dB = pt.islr_pslr(ant_pat, fs_bw_ratio=fs_bw_ratio, num_nulls_main=num_nulls_main, num_lobes=num_lobes, search_null=search_null)
    pslr_err = np.abs(pslr_dB - pslr_ideal_dB)
    islr_err = np.abs(islr_double_dB - islr_ideal_double_dB)

    npt.assert_array_less(pslr_err, pslr_max_err, 'PSLR error is larger than expected')
    npt.assert_array_less(islr_err, islr_max_err, 'ISLR error is larger than expected')
