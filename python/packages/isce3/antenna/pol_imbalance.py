"""
Functions and Classes for Polarimetric imabalance ratio of Antennas expressed
in antenna frame.
"""
from dataclasses import dataclass
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class PolImbalanceRatioAnt:
    """
    Transmit (TX) and Receive (RX) polarimetric channel ratio in elevation (EL)
    direction expressed in antenna frame.
    Pol imabalnce is defined as a complex ratio of "V co-pol" to "H co-pol".

    Attributes
    ----------
    tx_pol_ratio : scipy.interpolate.interp1d
        Complex polarimetric ratio (linear) on TX side extracted from V/H
        TX EL-cut antenna co-pol patterns as a function of EL angles in (rad).

    rx_pol_ratio : scipy.interpolate.interp1d
        Complex polarimetric ratio (linear) on RX side extracted from V/H
        RX EL-cut antenna co-pol patterns as a function of EL angles in (rad).

    """
    tx_pol_ratio: interp1d
    rx_pol_ratio: interp1d
