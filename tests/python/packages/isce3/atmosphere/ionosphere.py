import numpy as np
from osgeo import gdal

import isce3
from isce3.signal.interpolate_by_range import decimate_freq_a_array

from isce3.atmosphere.ionosphere_filter import IonosphereFilter, write_array
from isce3.atmosphere.main_band_estimation import (compute_unwrapp_error_main_diff_ms_band,
                                                   compute_unwrapp_error_main_side_band,
                                                   estimate_iono_main_side,
                                                   estimate_iono_main_diff)
from isce3.atmosphere.split_band_estimation import (compute_unwrapp_error_split_main_band,
                                                    estimate_iono_low_high)

def simulate_ifgrams(f, dr, dTEC):
    phi_non_dispersive = 4.0*np.pi*f*dr/isce3.core.speed_of_light
    K = 40.31
    phi_TEC = (-4.0*np.pi*K/(isce3.core.speed_of_light*f))*dTEC
    phi = phi_non_dispersive + phi_TEC

    return phi, phi_non_dispersive , phi_TEC

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

    phi_iono_LH, phi_n_LH = estimate_iono_low_high(f0=f0,
                                                   freq_low=f0L,
                                                   freq_high=f0H,
                                                   phi0_low=phi0L,
                                                   phi0_high=phi0H)

    phi_iono_ms, phi_n_ms = estimate_iono_main_side(f0=f0,
                                                    f1=f1,
                                                    phi0=phi0,
                                                    phi1=phi1)

    phi_iono_md, phi_n_md = estimate_iono_main_diff(f0=f0,
                                                    f1=f1,
                                                    phi0=phi0,
                                                    phi1=phi1)

    difference_ref_ls_abs = np.abs(phi0_iono - phi_iono_LH)
    difference_ref_ms_abs = np.abs(phi0_iono - phi_iono_ms)
    difference_ref_md_abs = np.abs(phi0_iono - phi_iono_md)

    assert difference_ref_ls_abs < 1e-5
    assert difference_ref_ms_abs < 1e-5
    assert difference_ref_md_abs < 1e-5

def test_unwrap_error_methods():
    f0 = 1.27e9
    f1 = f0 + 60.0e6
    BW = 20.0e6 # bandwidt
    f0L = f0 - BW/3.0
    f0H = f0 + BW/3.0

    dtecx = np.linspace(1.0*1e16, 2*1e16, 100)
    dtecy = np.linspace(1.0*1e16, 2*1e16, 100)
    dtecxy1, dtecxy2 = np.meshgrid(dtecx, dtecy)
    dTEC = dtecxy1 + dtecxy2

    phase = 0
    deltaR = (isce3.core.speed_of_light/f0/4.0/np.pi)*phase

    # simulate 2 dimensional phase for different center frequencies.
    phase0, _, _ = simulate_ifgrams(f0, deltaR, dTEC)
    phaseL, phi_low_non, phi_low_iono = simulate_ifgrams(f0L, deltaR, dTEC)
    phaseH, phi_high_non, phi_high_iono = simulate_ifgrams(f0H, deltaR, dTEC)
    phaseSideBand, phi_side_non, phi_side_iono = simulate_ifgrams(f1, deltaR, dTEC)

    # unwrap errors
    common_ref_err = np.zeros([100, 100])
    common_ref_err[40:60, 40:60] = common_ref_err[40:60, 40:60] - 2 * np.pi
    diff_ref_err = np.zeros([100, 100])
    diff_ref_err[50:100, 50:100] = diff_ref_err[50:100, 50:100] + 4 * np.pi

    # test for split_main_band
    phi_iono_lh, phi_n_lh = estimate_iono_low_high(f0, f0L, f0H, phaseL, phaseH)

    # add unwrapping errors to subbands
    phaseL_unwErr = phaseL.copy()
    phaseH_unwErr = phaseH.copy()

    phaseL_unwErr = phaseL_unwErr + common_ref_err
    phaseH_unwErr = phaseH_unwErr + common_ref_err
    phaseH_unwErr = phaseH_unwErr + diff_ref_err

    # assume that ionosphere phase is correctly estimated through filtering
    com_unw_err, diff_unw_err = compute_unwrapp_error_split_main_band(
        f0=f0,freq_low=f0L, freq_high=f0H,
        disp_array=phi_iono_lh, nondisp_array=phi_n_lh,
        low_sub_runw=phaseL_unwErr, high_sub_runw=phaseH_unwErr)

    difference_comref_ls_abs = np.sum(np.abs(com_unw_err * 2*np.pi - common_ref_err))
    difference_diffref_ls_abs = np.sum(np.abs(diff_unw_err* 2*np.pi - diff_ref_err))

    assert difference_comref_ls_abs < 1e-5
    assert difference_diffref_ls_abs < 1e-5

    # test for main_side_band
    phi_iono_ms, phi_n_ms = estimate_iono_main_side(f0, f1, phase0, phaseSideBand)

    # add unwrapping errors to subbands
    phaseSideBand_unwErr = phaseSideBand.copy()
    phase0_unwErr = phase0.copy()

    phase0_unwErr = phase0_unwErr + common_ref_err
    phaseSideBand_unwErr = phaseSideBand_unwErr + common_ref_err
    phaseSideBand_unwErr = phaseSideBand_unwErr + diff_ref_err

    # assume that ionosphere phase is correctly estimated through filtering
    com_unw_err, diff_unw_err = compute_unwrapp_error_main_side_band(
        f0=f0, f1=f1,
        disp_array=phi_iono_ms, nondisp_array=phi_n_ms,
        main_runw=phase0_unwErr, side_runw=phaseSideBand_unwErr)

    difference_comref_ms_abs = np.sum(np.abs(com_unw_err * 2*np.pi - common_ref_err))
    difference_diffref_ms_abs = np.sum(np.abs(diff_unw_err* 2*np.pi - diff_ref_err))

    assert difference_comref_ms_abs < 1e-5
    assert difference_diffref_ms_abs < 1e-5

    # test for main_diff_main_side_band
    phi_iono_md, phi_n_md = estimate_iono_main_diff(f0, f1, phase0, phaseSideBand)

    # assume that ionosphere phase is correctly estimated through filtering
    com_unw_err, diff_unw_err = compute_unwrapp_error_main_diff_ms_band(
        f0=f0, f1=f1,
        disp_array=phi_iono_ms, nondisp_array=phi_n_ms,
        main_runw=phase0_unwErr, side_runw=phaseSideBand_unwErr)

    difference_comref_md_abs = np.sum(np.abs(com_unw_err * 2*np.pi - common_ref_err))
    difference_diffref_md_abs = np.sum(np.abs(diff_unw_err* 2*np.pi - diff_ref_err))

    assert difference_comref_md_abs < 1e-5
    assert difference_diffref_md_abs < 1e-5

def test_ionosphere_filter():
    '''
    Check if filtered dispersive changes depending on iteration number
    The ionosphere script is designed to apply the filter only one time over
    valid region.
    '''
    # define kernel sizes in range and azimuth
    kernel_range_size = 40
    kernel_azimuth_size = 40
    kernel_sigma_range = 10
    kernel_sigma_azimuth = 10
    filter_iterations = 1
    filling_method = 'nearest'

    # initialize interferometric phase
    simul_disp_path = 'simulated_disp'
    simul_disp_sig_path = 'simulated_disp_sig'
    mask_path = 'mask_raster'
    filt_simul_disp_path = 'simulated_disp_filt'
    filt_simul_disp_sig_path = 'simulated_disp_sig_filt'
    f0 = 1.27e9
    dtecx = np.linspace(1.0*1e16, 2*1e16, 100)
    dtecy = np.linspace(1.0*1e16, 2*1e16, 100)
    dtecxy1, dtecxy2 = np.meshgrid(dtecx, dtecy)
    dTEC = dtecxy1 + dtecxy2
    phase = 0
    deltaR = (isce3.core.speed_of_light/f0/4.0/np.pi)*phase

    phase0, _, _ = simulate_ifgrams(f0, deltaR, dTEC)
    phase_sig = np.ones_like(phase0)

    # genate invalid regions
    maskraster = np.ones_like(phase0, dtype=bool)
    maskraster[50:70, 50:70] = 0

    # write data into files
    write_array(simul_disp_path,
        phase0, data_shape=np.shape(phase0))
    write_array(simul_disp_sig_path,
        phase_sig, data_shape=np.shape(phase_sig))
    write_array(mask_path,
        maskraster, data_shape=np.shape(maskraster))

    #initialize ionosphere filter object

    iono_filter_obj = IonosphereFilter(
        x_kernel=kernel_range_size,
        y_kernel=kernel_azimuth_size,
        sig_x=kernel_sigma_range,
        sig_y=kernel_sigma_azimuth,
        iteration=filter_iterations,
        filling_method=filling_method,
        outputdir='.')

    # run low pass filtering
    iono_filter_obj.low_pass_filter(
        input_data=simul_disp_path,
        input_std_dev=simul_disp_sig_path,
        mask_path=mask_path,
        filtered_output=filt_simul_disp_path,
        filtered_std_dev=filt_simul_disp_sig_path,
        lines_per_block=500)

    filt_iter_1_gdal = gdal.Open(filt_simul_disp_path)
    filt_iter_1 = filt_iter_1_gdal.ReadAsArray()
    filt_iter_1_gdal = None
    del filt_iter_1_gdal

    filter_iterations = 3
    iono_filter_obj = IonosphereFilter(
        x_kernel=kernel_range_size,
        y_kernel=kernel_azimuth_size,
        sig_x=kernel_sigma_range,
        sig_y=kernel_sigma_azimuth,
        iteration=filter_iterations,
        filling_method=filling_method,
        outputdir='.')
    iono_filter_obj.low_pass_filter(
        input_data=simul_disp_path,
        input_std_dev=simul_disp_sig_path,
        mask_path=mask_path,
        filtered_output=filt_simul_disp_path,
        filtered_std_dev=filt_simul_disp_sig_path,
        lines_per_block=500)

    filt_iter_3_gdal = gdal.Open(filt_simul_disp_path)
    filt_iter_3 = filt_iter_3_gdal.ReadAsArray()
    filt_iter_3_gdal = None
    del filt_iter_3_gdal
    # only compare the regions which is not affected by invalid region
    difference = np.sum(np.abs(filt_iter_1[:30, :30] - filt_iter_3[:30, :30] ))
    assert difference < 1e-5

def test_decimate_runw():
    main_slant = np.arange(500, 1000, 2)
    side_slant = np.arange(500, 1000, 4)
    main_runw = np.reshape(np.arange(500, 1000, 2) ,[1, -1])

    decimate_test = decimate_freq_a_array(main_slant,
        side_slant,
        main_runw)

    difference = np.sum(np.abs(decimate_test - side_slant))
    assert difference < 1e-5

if __name__ == '__main__':
    test_ionosphere_methods()
    test_unwrap_error_methods()
    test_ionosphere_filter()
    test_decimate_runw()
