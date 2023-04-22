"""
Functions and Classes for Polarimetric Cross-talk of Antennas expressed
in antenna frame.
"""
import numpy as np
import numpy.typing as npt
import typing
from dataclasses import dataclass


# Define datatype aliases
ArrayLikeComplex = typing.Union[complex, typing.Sequence[complex],
                                npt.NDArray[complex]]
VectorFloat = typing.Union[typing.Sequence[float], npt.NDArray[float]]


@dataclass
class CrossTalk:
    """
    Polarimetric Cross Talk Ratio Of Transmit (TX) and Receive (RX) Antennas.
    These parameters are used as part of polarimetric calibration in
    the polarimetric distortion matrix.

    Attributes
    ----------
    tx_xpol_h : complex scalar or 1-D array
        Cross polarization ratio for TX=H.
    tx_xpol_v : complex scalar or 1-D array
        Cross polarization ratio for TX=V
    rx_xpol_h : complex scalar or 1-D array
        Cross polarization ratio for RX=H
    rx_xpol_v : complex scalar or 1-D array
        Cross polarization ratio for RX=V
    el_ang : optional or 1-D array of float
        Elevation (EL) angles in radians defined in antenna frame.

    Raises
    ------
    ValueError
        Size mismatch among all parameters if `el_ang` is provided.

    Notes
    -----
    If `el_ang` is not None, then all parameters should be an array
    of the same size! Otherwise, a fixed EL-independent scalar value will be
    assumed by taking the mean among all values for each parameter.

    """
    tx_xpol_h: ArrayLikeComplex
    tx_xpol_v: ArrayLikeComplex
    rx_xpol_h: ArrayLikeComplex
    rx_xpol_v: ArrayLikeComplex
    el_ang: typing.Optional[VectorFloat] = None

    def __post_init__(self):
        self.tx_xpol_h = np.asarray(self.tx_xpol_h)
        self.tx_xpol_v = np.asarray(self.tx_xpol_v)
        self.rx_xpol_h = np.asarray(self.rx_xpol_h)
        self.rx_xpol_v = np.asarray(self.rx_xpol_v)
        if self.el_ang is None:
            # a constant value over entire swath
            self.tx_xpol_h = self.tx_xpol_h.mean()
            self.tx_xpol_v = self.tx_xpol_v.mean()
            self.rx_xpol_h = self.rx_xpol_h.mean()
            self.rx_xpol_v = self.rx_xpol_v.mean()
        else:
            # el-varying values over entire swath
            self.el_ang = np.asarray(self.el_ang)
            size_el = self.el_ang.size
            if not (self.tx_xpol_h.size == size_el and
                    self.tx_xpol_v.size == size_el and
                    self.rx_xpol_h.size == size_el and
                    self.rx_xpol_v.size == size_el):
                raise ValueError('Size mismtach!')
