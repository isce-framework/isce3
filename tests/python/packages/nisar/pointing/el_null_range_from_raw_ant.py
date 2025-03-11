import iscetest
from nisar.pointing import el_null_range_from_raw_ant
from nisar.products.readers.Raw import Raw
from isce3.geometry import DEMInterpolator
from nisar.products.readers.antenna import AntennaParser
from isce3.core import speed_of_light, TimeDelta

import numpy.testing as npt
import numpy as np
import os


def test_el_null_range_from_raw_ant():
    # Inputs:

    # filename of L0B (multi-channel echo or DM2) and ANT (multi-beam)
    l0b_file = 'REE_L0B_CHANNEL4_EXTSCENE_PASS1_LINE3000_CALIB.h5'
    ant_file = 'REE_ANTPAT_CUTS_BEAM4.h5'
    az_block_dur = 0.75  # (sec)
    txrx_pol = 'VV'
    sample_delays = np.zeros(3, dtype=int)
    imbalances = np.ones(3, dtype='c8')

    # build object and scalar used for validations:

    # build DEM object
    dem_interp_obj = DEMInterpolator()
    # parse L0B to get orbit, attitude, echo data
    raw_obj = Raw(hdf5file=os.path.join(iscetest.data, l0b_file))
    # parse antenna EL cut
    ant_obj = AntennaParser(os.path.join(iscetest.data, ant_file))
    # get slant range
    sr_lsp = raw_obj.getRanges('A', txrx_pol[0])
    # wavelength
    center_freq = raw_obj.getCenterFrequency('A', txrx_pol[0])
    wavelength = speed_of_light / center_freq
    # echo ref and azimuth time
    ref_utc_echo, aztime_echo = raw_obj.getPulseTimes('A', txrx_pol[0])
    azt_first = ref_utc_echo + TimeDelta(aztime_echo[0])
    azt_last = ref_utc_echo + TimeDelta(aztime_echo[-1])
    # get shape of raw data
    raw_dset = raw_obj.getRawDataset('A', txrx_pol)
    num_channels, num_rgls, num_rgbs = raw_dset.shape

    # get number of beams
    num_beams = ant_obj.num_beams(txrx_pol[1])
    num_nulls = num_beams - 1
    # calculate number of azimuth blocks
    prf = raw_obj.getNominalPRF('A', txrx_pol[0])
    num_azimuth_block = num_rgls // round(az_block_dur * prf)
    # expected output array size
    array_size = num_azimuth_block * num_nulls
    # get peak location for each beam
    el_peak_loc = np.zeros(num_beams)
    for nn in range(num_beams):
        beam = ant_obj.el_cut(nn + 1, txrx_pol[1])
        el_peak_loc[nn] = beam.angle[abs(beam.copol_pattern).argmax()]
    # get approximate null locations (half way between adjacent peaks)
    el_null_loc = np.rad2deg(0.5 * (el_peak_loc[:-1] + el_peak_loc[1:]))

    # call the function and validate its output shape, range of values, etc:
    (null_num, sr_echo, el_ant, mag_ratio, az_dt, null_flag, mask_valid,
     pol, wvl) = el_null_range_from_raw_ant(
         raw_obj, ant_obj, dem_interp=dem_interp_obj, freq_band='A',
         txrx_pol=txrx_pol, az_block_dur=az_block_dur,
         sample_delays_wrt_left=sample_delays,
         imbalances_right2left=imbalances
    )

    # Validate output size
    npt.assert_equal(null_num.size, array_size,
                     err_msg='Wrong size of null numbers')
    npt.assert_equal(sr_echo.size, array_size,
                     err_msg='Wrong size of slant ranges')
    npt.assert_equal(el_ant.size, array_size,
                     err_msg='Wrong size of antenna EL angles')
    npt.assert_equal(mag_ratio.size, array_size,
                     err_msg='Wrong size of magnitude ratio')
    npt.assert_equal(az_dt.size, array_size,
                     err_msg='Wrong size of azimuth datetime tags')
    npt.assert_equal(null_flag.size, array_size,
                     err_msg='Wrong size of null convergence flags')
    npt.assert_equal(mask_valid.size, array_size,
                     err_msg='Wrong size of valid mask of null')

    # Validate values
    npt.assert_equal(pol, txrx_pol, err_msg='Wrong TxRx Pol')
    npt.assert_allclose(wvl, wavelength, err_msg='Wrong wavelength')
    npt.assert_equal(np.all(null_flag), True,
                     err_msg='Some null flags are False')
    npt.assert_equal(np.all(mask_valid), True,
                     err_msg='Some valid mask values are False')
    # loop over azimuth blocks
    for cc in range(num_azimuth_block):
        block_slice = slice(cc * num_nulls, (cc + 1) * num_nulls)
        # validate min, max and ascending order of slant ranges
        npt.assert_equal(
            np.all(sr_echo[block_slice] > sr_lsp.first), True,
            err_msg=f'Some slant ranges for block # {cc} are too small!'
        )
        npt.assert_equal(
            np.all(sr_echo[block_slice] < sr_lsp.last), True,
            err_msg=f'Some slant ranges for block # {cc} are too large!'
        )
        npt.assert_equal(
            np.all(np.diff(sr_echo[block_slice]) > 0), True,
            err_msg=(f'Slant ranges for block # {cc} are not monotonically '
                     'increasing')
        )
        # validate magnitude ratio of nulls per each block
        is_mag_ratio_valid = (np.all(mag_ratio[block_slice] > 0) and
                              np.all(mag_ratio[block_slice] < 1))
        npt.assert_equal(
            is_mag_ratio_valid, True,
            err_msg=f'Mag ratio is out of range ]0,1[ for block # {cc}'
        )
        # Validate azimuth time tag to be within expected range
        npt.assert_equal(
            np.all(az_dt[block_slice] > azt_first), True,
            err_msg=f'Some azimuth timedate for block # {cc} are too small!'
        )
        npt.assert_equal(
            np.all(az_dt[block_slice] < azt_last), True,
            err_msg=f'Some azimuth timedate for block # {cc} are too large!'
        )
        dif_azdt = np.asarray([val.seconds for val in
                               np.diff(az_dt[block_slice])])
        npt.assert_equal(
            np.all(dif_azdt >= 0), True,
            err_msg=(f'Azimuth datetimes for block # {cc} are not '
                     'monotonically increasing')
        )

        # loop over nulls per block
        for nn in range(num_nulls):
            idx = cc * num_nulls + nn
            # validate EL angle
            npt.assert_allclose(
                el_ant[idx], el_null_loc[nn], atol=0.1,
                err_msg=f'Wrong EL angle for block # {cc} and null # {nn}'
            )
