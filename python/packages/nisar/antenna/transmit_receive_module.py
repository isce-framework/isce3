from typing import Sequence
from dataclasses import dataclass
from enum import IntEnum, unique


@unique
class CalPath(IntEnum):
    """Enumeration for type of TRM (trasnmit-receive module) calibration path
    containing transmit chirp.

    There are three types of calibration: HCAL=HPA, LCAL=LNA, and BCAL=BYPASS.
    See [1]_ for details.

    References
    ----------
    .. [1] H. Ghaemi, "DSI SweepSAR On-Board DSP Algorithms Description ,"
        JPL D-95646, Rev 14, 2018.

    """
    HPA = 0
    LNA = 1
    BYPASS = 2


@dataclass(frozen=True)
class TxTrmInfo:
    """Transmit (TX) side info of transmit-receive module (TRM) class used to
    compute the beamformed (BMF) Tx antenna pattern in elevation (EL)
    direction.

    See [1]_ for technical description of transmit-path parameters and
    its application in transmit-path calibration.

    Attributes
    ----------
    time : sequence of float
        Pulse time of range line in seconds w.r.t radar product reference epoch
    channels : sequence of int
        Active Tx channel numbers defined by L0B raw data 'listOfTxTRMs'.
        The values shall be equal or greater than 1.
    correlator_tap2 : 2D sequence of complex
        second tap of 3-tap correlator for all range lines and channels
        with shape = (rangelines, total TX channels)
    cal_path_mask : sequence of CalPath
        Enum Mask from L0B data which indicates range line types:
        0 = HCAL, 1 = LCAL, 2 = BCAL
    tx_phase : np.ndarray(float), optional
        2-D array of phases of TX paths in radians with shape
        (rangelines, total TX channels). If None, the phase extracted from
        `correlator_tap2` will be used on TX side.

    References
    ----------
    .. [1] H. Ghaemi, "DSI SweepSAR On-Board DSP Algorithms Description ,"
        JPL D-95646, Rev 14, 2018.

    """
    time: Sequence[float]
    channels: Sequence[int]
    correlator_tap2: Sequence[Sequence[complex]]
    cal_path_mask: Sequence[CalPath]
    tx_phase: Sequence[Sequence[float]] = None


@dataclass(frozen=True)
class RxTrmInfo:
    """Receive (RX) side info of transmit-receive module (TRM) class used to
    compute the digitally beamformed (DBF) RX antenna pattern in elevation
    (EL) direction.

    See [1]_ for technical description of DBF-related parameters and algorithm.

    Attributes
    ----------
    time : sequence of float
        time tag of range lines in seconds w.r.t radar product reference epoch
    channels : sequence of int
        Active Rx channel numbers defined by L0B raw data 'listOfRxTRMs'.
        The values shall be equal or greater than 1.
    rd : sequence of int
        Round trip time at starting elevation angle in Time-To-Angle
        LUT calculated with respect to transmit pulse. Values are provided
        at 'fs_win' clock sample counts with size equal to total number of
        RX channels. This value along with `wd` determines data window
        position (DWP) for each channel.
    wd : sequence of int
        Receive window delay to first valid data sample relative to RD.
        Value is provided as 'fs_win' clock sample counts.
        The size is total number of RX channels.
    wl : sequence of int
        Length of receive data window provided as 'fs_win' clock sample counts.
        The size is total number of RX channels.
    ac_dbf_coef : 2D sequence of complex
        angle-to-coeff tables for all channels with shape
        (total RX channels, number of coeffs)
    ta_dbf_switch : 2-D sequence of int
        'N' time-index values @ 'fs_ta' clock rate to map fast-time
        indices to 'N' Elevation angle indices for each channel.
        The array shape is (total number of channels, number of angle indexes)
    el_ang_dbf : 2-D sequence of float
        Uniformly-sampled elevation (EL) angles corresponds to DBF coefficients
        (AC table) in radians with shape
        (total RX channels, number of DBF coeffs).
    fs_win : float, default=240e6
        Sampling frequency in (Hz) at which fast-time range window paramters
        RD/WD/WL are generated.
    fs_ta : float, default=96e6
        Sampling rate of DBF fast-time indexing in time-to-angle and
        angle-to-coefficient transformation, in Hertz. DBF is done at this
        sampling rate. This value can be different from sampling frequency
        of Raw data and clock rate for fast-time range gating of Raw data.

    References
    ----------
    .. [1] H. Ghaemi, "DSI SweepSAR On-Board DSP Algorithms Description ,"
        JPL D-95646, Rev 14, 2018.

    """
    time: Sequence[float]
    channels: Sequence[int]
    rd: Sequence[int]
    wd: Sequence[int]
    wl: Sequence[int]
    ac_dbf_coef: Sequence[Sequence[complex]]
    ta_dbf_switch: Sequence[Sequence[int]]
    el_ang_dbf: Sequence[Sequence[float]]
    fs_win: float = 240e6
    fs_ta: float = 96e6
