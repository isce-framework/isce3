import iscetest
import numpy as np
import numpy.testing as npt
import os
from nisar.products.readers.antenna import AntennaParser
from isce3.geometry import DEMInterpolator as DEMInterp
from isce3.core import Ellipsoid, Quaternion
from nisar.products.readers.Raw import Raw
from isce3 import antenna as ant
from nisar import antenna as bf_ant
import h5py
from collections import namedtuple
from enum import IntEnum, unique
from .instrument_parser import InstrumentParser
import warnings


def get_test_file():
    data_file = os.path.join(iscetest.data, "bf", "REE_L0B_ECHO_DATA.h5")
    ant_file = os.path.join(iscetest.data, "bf", "REE_ANTPAT_CUTS_DATA.h5")
    instrument_table_file = os.path.join(
        iscetest.data, "bf", "instrumentTables_20220207.h5"
    )

    return ant_file, data_file, instrument_table_file


def parse_ant(ant_file, pol):
    """
    Parse antenna elevation gain pattern
    Return elevation gain, azimuth cut angle and antenna gain for all beams.

    Parameters:
    -----------
    ant_file: str
        antenna gain H5 file
    pol: str
        Tx polarity H or V used to select antenna gain from antenna H5 file

    Returns:
    --------
    el_gain_array: 2D array of complex
        antenna elevation gain, dim = [# of chan x # of angels]
    el_angles_rad: array of float
        antenna elevation angles in radian
    az_cut_angles_rad: float
        scalar angle value of azimuth cut angle in radian
    """

    ant_parsed = AntennaParser(ant_file)
    num_beams = ant_parsed.num_beams(pol)

    el_cut_first_beam = ant_parsed.el_cut(1, pol)
    az_cut_angles_rad = el_cut_first_beam.cut_angle
    el_angles_rad = el_cut_first_beam.angle
    num_angles = len(el_angles_rad)

    el_gain_array = np.zeros((num_beams, num_angles), dtype=complex)
    for beam in range(num_beams):
        el_cut_beam = ant_parsed.el_cut(beam + 1, pol)
        el_gain_array[beam] = el_cut_beam.copol_pattern

    ElAntPattern = namedtuple(
        "ElAntPattern", "el_ant_gain, el_angles_rad, az_cut_angles_rad"
    )
    el_ant_pattern = ElAntPattern(el_gain_array, el_angles_rad, az_cut_angles_rad)

    return el_ant_pattern


def read_raw_data(data_file, freq_group, pols):
    """
    Parse raw L0B file

    Parameters:
    -----------
    data_file: str
        Raw L0B file
    freq_group: {'A', 'B'}
        L0B raw data frequency band char selection 'A' or 'B'
    pols: {'HH', 'HV', 'VH', 'VV'}
        L0B raw data file Tx and Rx pol selection 'HH', 'HV', 'VH', or 'VV'

    Returns:
    --------
    orbit: obj
        ISCE3 data orbit object
    attitude: obj
        ISCE3 data attitude object
    transmit_time: array of float
        transmit time of range line in seconds w.r.t radar product reference
    slant_range: isce3.core.Linspace object
        data slant range bins
    fc: float
        data center frequency
    list_tx_trm: array of int
        Tx Modules activated for beamforming
    list_rx_trm: array of int
        Rx Modules activated for beamforming
    rng_lines_idx: array of int
        range line indices
    cal_path_mask: int Enum
        Calibration path mask generated for all range line
        HPA = 0, LNA = 1, and BYPASS = 2
    correlator_tap2: 2D array of complex
        second tap of 3-tap correlator for all range lines
    """

    raw = Raw(hdf5file=data_file)
    orbit = raw.getOrbit()
    attitude = raw.getAttitude()
    transmit_time = raw.getPulseTimes(freq_group, tx=pols[0])[-1]
    fc = raw.getCenterFrequency(freq_group)
    slant_range = raw.getRanges(freq_group, pols[0])

    list_tx_trm = raw.getListOfTxTRMs(freq_group, pols[0])
    list_rx_trm = raw.getListOfRxTRMs(freq_group, pols)
    rng_lines_idx = raw.getRangeLineIndex(freq_group, pols[0])
    cal_path_mask = raw.getCalType(freq_group, tx=pols[0])
    correlator_3taps = raw.getChirpCorrelator(freq_group, pols[0])
    correlator_tap2 = correlator_3taps[..., 1]

    return (
        orbit,
        attitude,
        transmit_time,
        slant_range,
        fc,
        list_tx_trm,
        list_rx_trm,
        rng_lines_idx,
        cal_path_mask,
        correlator_tap2,
    )


def parse_instrument_table(cal_lut_file, pol):
    """
    Parse instrument table, including Angle to Coef. and Time to Angle tables

    Parameters:
    -----------
    cal_lut_file: str
        instrument table H5 file
    pol: str
        Rx polarity H or V used to select antenna gain from antenna H5 file

    Returns:
    --------
    ac_angle_count: int
        Angle to Coeff. Table number of angles
    ac_chan_coef: 2D array of complex
        Rx CAL path channel coeff., size = [num of chan x 256]
    angle_low_idx: float
        start angle for channel coeff. application
    angle_high_idx: float
        stop angle for channel coeff. application
    ta_dbf_switch: array of int
        Time to Angle Table entries
    """

    h5 = InstrumentParser(cal_lut_file)

    # Angle to Coef. Table: Angle count
    ac_angle_count = h5.get_ac_angle_count(pol)

    # Angle to Coef. Table: AC Coef.
    ac_chan_coef = h5.get_ac_coef(pol)

    # Time to Angle Table: DBF Switch
    ta_dbf_switch = h5.get_ta_dbf_switch(pol)

    return ac_angle_count, ac_chan_coef, ta_dbf_switch


def test_beamform():
    # Test parameters
    pols = "HH"
    freq_group = "A"
    dem_hgt = 0
    dem = DEMInterp(dem_hgt)

    # rd, wd, wl @ 240 MHz
    rd = np.array(
        [
            1387305,
            1387305,
            1387305,
            1387305,
            1446925,
            1446925,
            1446925,
            1446925,
            1536965,
            1536965,
            1536965,
            1536965,
        ]
    )
    wd = np.array(
        [4670, 4670, 4670, 46750, 5670, 25285, 47100, 69115, 2390, 28350, 54650, 83960]
    )
    wl = np.array(
        [
            42080,
            60620,
            80235,
            59970,
            63445,
            67145,
            71290,
            75575,
            81570,
            116950,
            90650,
            61340,
        ]
    )

    # Tx Parameters
    # Constant phase offset based on temperature for each channel
    tx_correction_factor = np.ones(12, dtype=complex)

    # Rx Correction factor
    rx_correction_factor = np.ones(12, dtype=complex)

    # Raw and Antenna file path
    ant_file, data_file, instrument_table_file = get_test_file()

    # Parse Antenna gain
    el_ant_pattern = parse_ant(ant_file, pols[0])

    # Parse Raw data
    (
        orbit,
        attitude,
        transmit_time,
        slant_range,
        fc,
        list_tx_trm,
        list_rx_trm,
        rng_lines_idx,
        cal_path_mask,
        correlator_tap2,
    ) = read_raw_data(data_file, freq_group, pols)

    # Rx: Parse instrument table for Rx weights
    (
        rx_ac_angle_count,
        rx_ac_chan_coef,
        rx_ta_dbf_switch,
    ) = parse_instrument_table(instrument_table_file, pols[1])

    # RX Parameters
    # TA table sampling frequency
    dbf_fs = 96e6
    adc_fs = 240e6
    dbf_switch_fs = 48e6
    num_chan = len(list_rx_trm)
    num_chan_qfsp = 4
    pulse_time = transmit_time[1]

    # Instantiate antenna object
    tx_trm, rx_trm = bf_ant.get_tx_and_rx_trm_info(
        transmit_time,
        fc,
        list_tx_trm,
        list_rx_trm,
        correlator_tap2,
        cal_path_mask,
        dbf_fs,
        adc_fs,
        dbf_switch_fs,
        rd,
        wd,
        wl,
        rx_ac_chan_coef,
        rx_ta_dbf_switch,
        rx_ac_angle_count,
        num_chan_qfsp,
        rx_correction_factor,
        tx_correction_factor,
    )

    # Instantiate Beamformer object
    beamformer = bf_ant.ElevationBeamformer(
        orbit, attitude, dem, slant_range, el_ant_pattern, tx_trm, rx_trm
    )

    # Determine combined Tx/Rx gain pattern based on pulse time
    (
        bmf_sr_interp,
        bmf_sr_interp_tx,
        bmf_sr_interp_rx,
        slant_range_el,
    ) = beamformer.form_two_way(pulse_time)

    # Pass/Fail
    #Rx/Tx Pwr difference margin in dB
    rx_dbf_pwr_tol_db = 0.4
    tx_bmf_pwr_tol_db = 0.4

    #RMS residual margin = 1 deg
    rx_dbf_res_phase_rms_tol_deg = 1.0

    # Read REE Rx DBF and TX BMF gain patterns for verification
    rx_dbf_path = f"/RX_DBF_{pols[1]}/elevation/copol_pattern"
    tx_bmf_path = f"/TX_BMF_{pols[0]}/elevation/copol_pattern"

    fid = h5py.File(ant_file, "r", libver="latest", swmr=True)
    ree_rx_dbf = np.asarray(fid[rx_dbf_path])
    ree_tx_bmf = np.asarray(fid[tx_bmf_path])

    # Interpolate REE Rx DBF gain from angle to slant_range domain
    slant_range_data_fs = np.asarray(slant_range)
    ree_rx_dbf_slant_range = np.interp(slant_range_data_fs, slant_range_el, ree_rx_dbf)

    # Interpolate REE Tx BMF gain from angle to slant_range domain
    ree_tx_bmf_slant_range = np.interp(slant_range_data_fs, slant_range_el, ree_tx_bmf)

    # Difference between REE and RX DBF
    amp2pwr_db = lambda x: 20 * np.log10(np.abs(x))
    rx_dbf_pwr_ratio_db = np.squeeze(
        amp2pwr_db(np.abs(bmf_sr_interp_rx) / np.abs(ree_rx_dbf_slant_range))
    )
    tx_bmf_pwr_ratio_db = np.squeeze(
        amp2pwr_db(np.abs(bmf_sr_interp_tx) / np.abs(ree_tx_bmf_slant_range))
    )

    # Apply -3dB mask to only compare swath of interest
    # Use REE BMF patterns as benchmarks.
    # Rx Mask
    ree_rx_dbf_pwr_db = amp2pwr_db(ree_rx_dbf_slant_range)
    mask_rx_3db = ree_rx_dbf_pwr_db > (ree_rx_dbf_pwr_db.max() - 3)
    rx_dbf_pwr_diff_max_db = np.amax(np.abs(rx_dbf_pwr_ratio_db[mask_rx_3db]))

    # Tx Mask
    ree_tx_bmf_pwr_db = amp2pwr_db(ree_tx_bmf_slant_range)
    mask_tx_3db = ree_tx_bmf_pwr_db > (ree_tx_bmf_pwr_db.max() - 3)
    tx_bmf_pwr_diff_max_db = np.amax(np.abs(tx_bmf_pwr_ratio_db[mask_tx_3db]))

    # Rx DBF residual phase
    num_rng_bins = len(bmf_sr_interp_rx[0])
    bmf_sr_interp_rx_phase_res = np.angle(bmf_sr_interp_rx, deg=True)

    # Compute Rx DBF residual phase RMS
    rx_bmf_res_phase_rms = np.sqrt(
        np.sum(bmf_sr_interp_rx_phase_res ** 2) / num_rng_bins
    )

    # Compare Tx and Rx beamforming gain pattern power against those of REE
    npt.assert_array_less(
        rx_dbf_pwr_diff_max_db,
        rx_dbf_pwr_tol_db,
        "Rx DBF error is larger than expected",
    )
    npt.assert_array_less(
        tx_bmf_pwr_diff_max_db,
        tx_bmf_pwr_tol_db,
        "Tx BMF error is larger than expected",
    )
    npt.assert_array_less(
        rx_bmf_res_phase_rms,
        rx_dbf_res_phase_rms_tol_deg,
        "Rx DBF residual phase error is larger than expected",
    )


if __name__ == "__main__":
    test_beamform()
