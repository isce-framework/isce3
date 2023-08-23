"""
Functions and Classes for Polarimetric Cross-talk of Antennas expressed
in antenna frame.
"""
from dataclasses import dataclass
from scipy.interpolate import interp1d


@dataclass
class CrossTalk:
    """
    Polarimetric Cross Talk Ratio Of Transmit (TX) and Receive (RX) Antennas
    in elevation (EL) direction.

    The cross-talk complex values are expressed as a function of EL angles in
    the form of 1-D LUT via scipy.interpolate.interp1d.
    Ideally, the cross talk values are assumed to be azimuth-integrated ratio
    of cross-pol  to co-pol radiation patterns. However, one may use EL-cuts
    at a fixed azimuth angle. In either way, that implies cross talk values are
    simply functions of EL angles without any azimuth angle dependency.

    These parameters are used as part of polarimetric calibration in
    the polarimetric distortion matrix.

    Attributes
    ----------
    tx_xpol_h : scipy.interpolate.interp1d
        Complex cross polarization ratio for TX=H as a function of EL in (rad).
    tx_xpol_v : scipy.interpolate.interp1d
        Complex cross polarization ratio for TX=V as a function of EL in (rad).
    rx_xpol_h : scipy.interpolate.interp1d
        Complex cross polarization ratio for RX=H as a function of EL in (rad).
    rx_xpol_v : scipy.interpolate.interp1d
        Complex cross polarization ratio for RX=V as a function of EL in (rad).

    """
    tx_xpol_h: interp1d
    tx_xpol_v: interp1d
    rx_xpol_h: interp1d
    rx_xpol_v: interp1d
