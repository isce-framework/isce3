import numpy as np
from typing import Sequence
from dataclasses import dataclass, field
from enum import IntEnum, EnumMeta, unique

def default_correction_factor():
    return np.ones(12, dtype=complex)

@unique
class CalPath(IntEnum):
    """Tx calibration range line types (HCAL = HPA, LCAL = LNA, BCAL = BYPASS)"""

    HPA = 0
    LNA = 1
    BYPASS = 2


@dataclass
class TxTrmInfo:

    """TX Antenna Pattern class used to compute the Tx normalized beamformed gain pattern.

    Attributes:
    -----------------
    transmit_time: sequence of float
        transmit pulse time of range line in seconds w.r.t radar product reference
    fc: float
        Radar center frequency, in Hertz
    channels: sequence of int
        Tx channels defined by L0B raw data 'listOfTxTRMs'
    correlator_tap2: 2D sequence of complex
        second tap of 3-tap correlator for all range lines and channels
    cal_path_mask: sequence of CalPath
        Enum Mask from L0B data which indicates range line types: 0 = HCAL, 1 = LCAL, 2 = BCAL
    tx_correction_factor: sequence of complex
        Tx Cal correction factor that captures both magnitude and phase variation due to
        temperature. A place holder initialied to one is designated in case if
        temperature calibration is applied, len = num of channels
    """

    transmit_time: Sequence[float]
    fc: float
    channels: Sequence[int]
    correlator_tap2: Sequence[Sequence[complex]]
    cal_path_mask: Sequence[CalPath]
    tx_correction_factor: Sequence[complex] = field(default_factory=default_correction_factor)


@dataclass
class RxTrmInfo:

    """Rx Antenna Pattern class used to compute the Rx normalized beamformed gain pattern.

    Attributes:
    -----------------
    transmit_time: sequence of float
        transmit time of range line in seconds w.r.t radar product reference
    fc: float
        Radar center frequency, in Hertz
    channels: sequence of int
        Rx channels defined by L0B raw data 'listOfRxTRMs'
    rd: sequence of int
        Round trip time at starting elevation angle in Time-To-Angle LUT calculated with respect to a
        transmit pulse. Value is provided as 'adc_fs' clock sample counts, len = 12
    wd: sequence of int
        Receive window delay to first valid data sample relative to RD.
        Value is provided as 'adc_fs' clock sample counts, len = 12
    wl: sequence of int
        Length of receive data window provided as 'adc_fs' clock sample counts, len = 12
    ac_chan_coef: 2D sequence of complex
        Rx CAL path channel coeff., size = [num of chan x num of coeffs]
    ta_dbf_switch: sequence of int
        'N' time-index values @ 'ta_lut_fs' clock rate to map fast-time indices to 'N' Elevation
         angle indices for each channel
    num_lut_items: int
        number of entries in CAL Angle-To-Coefficient and Time-To-Angle tables
    num_chan_qfsp: int
        There are 4 channels/beams being processed during beamforming per qFSP FPGA.
    dbf_fs: float
        Sampling frequency of DBF fast-time indexing in time-to-angle and angle-to-coefficient
        transformation, in Hertz.  This value can be different from sampling frequency of Raw data
        and clock rate for fast-time range gating of Raw data.
    adc_fs: float
        ADC sampling frequency at which fast-time range gating paramters RD/WD/WL are generated.
    ta_lut_fs: float
        sampling frequency of Time to Angle Table dbf_switch values.
        NISAR: number of 96-MHz clock rate divided by 2, which is 48 MHz.
    rx_correction_factor: sequence of complex
        A place holder is designated for applying possbile secondary Rx phase and magnitude corrections,
        currently initialized to one, 1 per each channel, size = num_chan
    """

    transmit_time: Sequence[float]
    fc: float
    channels: Sequence[int]
    rd: Sequence[int]
    wd: Sequence[int]
    wl: Sequence[int]
    ac_chan_coef: Sequence[Sequence[complex]]
    ta_dbf_switch: Sequence[int]
    num_lut_items: int = 256
    num_chan_qfsp: int = 4
    dbf_fs: float = 96e6
    adc_fs: float = 240e6
    ta_lut_fs: float = 48e6
    rx_correction_factor: Sequence[complex] = field(default_factory=default_correction_factor)


def get_tx_and_rx_trm_info(
    transmit_time,
    fc,
    channels_tx,
    channels_rx,
    correlator_tap2,
    cal_path_mask,
    dbf_fs,
    adc_fs,
    ta_lut_fs,
    rd,
    wd,
    wl,
    ac_chan_coef,
    ta_dbf_switch,
    num_lut_items=256,
    num_chan_qfsp=4,
    rx_correction_factor=field(default_factory=default_correction_factor),
    tx_correction_factor=field(default_factory=default_correction_factor),
):

    """This function instantiates TxTrmInfo and RxTrmInfo objects.

    Attributes:
    -----------
    transmit_time: sequence of float
        transmit time of range line in seconds w.r.t radar product reference
    fc: float
        Radar center frequency, in Hertz
    channels_tx: sequence of int
        Tx channels defined by L0B raw data 'listOfTxTRMs'
    channels_rx: sequence of int
        Rx channels defined by L0B raw data 'listOfRxTRMs'
    correlator_tap2: 2D sequence of complex
        second tap of 3-tap correlator for all range lines
    cal_path_mask: sequence of CalPath
        Enum Mask from L0B data which indicates range line types: 0 = HCAL, 1 = LCAL, 2 = BCAL
    dbf_fs: float
        Time to Angle Table Sampling frequency,in hertz, may be different from Raw data frequency
    adc_fs: float
        ADC sampling frequency. It is also the frequency which RD/WD/WL are generated.
    ta_lut_fs: float
        sampling frequency of Time to Angle Table dbf_switch values.
        NISAR: number of 96-MHz clock rate divided by 2, which is 48 MHz.
    rd: sequence of int
        Round trip time at starting elevation angle in Time-To-Angle LUT calculated with respect to a
        transmit pulse. Value is provided as 240 MHz clock sample counts, len = 12
    wd: sequence of int
        Receive window delay to first valid data sample relative to RD.
        Value is provided as 240 MHz clock sample counts, len = 12
    wl: sequence of int
        Length of receive data window provided as 240 MHz clock sample counts, len = 12
    ac_chan_coef: 2D sequence of complex
        Rx CAL path channel coeff., size = [num of chan x num of Angle-To-Coefficient table coeff.]
    ta_dbf_switch: sequence of int
        'N' time-index values @ 'ta_lut_fs' clock rate to map fast-time indices to 'N' Elevation
         angle indices for each channel
    num_lut_items: int
        number of entries in CAL Angle-To-Coefficient and Time-To-Angle tables
    num_chan_qfsp: int
        number of channels/beams being processed during beamforming per qFSP FPGA, default=4
    rx_correction_factor: sequence of complex
        A place holder is designated for applying possbile secondary Rx phase and magnitude correction,
        currently initialized to one, 1 per each channel, size = num_chan
    tx_correction_factor: sequence of complex
        Tx Cal correction factor that captures both magnitude and phase variation due to
        temperature. A place holder initialied to one is designated in case if
        temperature calibration is applied, len = num of channels
    """

    # Instantiate Tx Beamformer object
    tx_trm_info = TxTrmInfo(
        transmit_time,
        fc,
        channels_tx,
        correlator_tap2,
        cal_path_mask,
        tx_correction_factor,
    )

    # Instantiate Rx Beamformer object
    rx_trm_info = RxTrmInfo(
        transmit_time,
        fc,
        channels_rx,
        rd,
        wd,
        wl,
        ac_chan_coef,
        ta_dbf_switch,
        num_lut_items,
        num_chan_qfsp,
        dbf_fs,
        adc_fs,
        ta_lut_fs,
        rx_correction_factor,
    )

    return tx_trm_info, rx_trm_info
