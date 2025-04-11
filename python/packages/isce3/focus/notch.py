from dataclasses import dataclass
from enum import Enum, unique
import logging

log = logging.getLogger("isce3.focus.notch")


@unique
class FrequencyDomain(str, Enum):  # StrEnum in Python 3.11+
    """
    Flags denoting reference of a given frequency value.
    """

    BASEBAND = "baseband"
    """
    The frequency is relative to the nominal center frequency of the band.
    """

    RADIO_FREQUENCY = "radio_frequency"
    """The frequency is referenced to an absolute frequency of 0"""


@dataclass(frozen=True)
class Notch:
    """
    Frequency-domain notch filter parameters

    Parameters
    ----------
    frequency : float
        Center frequency of notch filter
    bandwidth : float
        Width of notch filter, same units as frequency
    domain : FrequencyDomain
        Flag indicating whether frequency is referenced to baseband or RF.
    """
    frequency: float
    bandwidth: float
    domain: FrequencyDomain

    def normalized(self, fs: float, fc: float = 0.0):
        """
        Shift notch to baseband and normalize by sample rate, as needed by
        isce3.focus.RangeComp.apply_notch.

        Parameters
        ----------
        fs : float
            Sample rate of digital signal
        fc : float, optional
            Carrier frequency of RF signal (if applicable)

        Returns
        -------
        notch : Notch
            Normalized notch filter parameters
        """
        if self.domain == FrequencyDomain.RADIO_FREQUENCY:
            if fc == 0.0:
                log.warning("RF Notch was normalized with fc=0")
            f = (self.frequency - fc) / fs
        else:
            f = self.frequency / fs
        b = self.bandwidth / fs
        return Notch(f, b, "baseband")
