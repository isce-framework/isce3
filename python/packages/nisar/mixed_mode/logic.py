from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from nisar.products.readers.Raw.Raw import RawBase
from typing import Set


def null_band():
    return Band(0.0, 0.0)


@dataclass(frozen=True)
class Band:
    """
    A radio frequency band

    Attributes
    ----------
    center : float
        Center frequency in Hz
    width : float
        Band width in Hz
    """
    center: float
    width: float

    def __post_init__(self):
        if self.width < 0.0:
            raise ValueError("Band.width cannot be negative")

    @property
    def low(self) -> float:
        "Lower edge of band in Hz"
        return self.center - self.width * 0.5

    @property
    def high(self) -> float:
        "Upper edge of band in Hz"
        return self.center + self.width * 0.5

    def intersection(self, other: 'Band') -> 'Band':
        """
        Returns overlapping portion of two frequency bands.
        If the bands do not overlap then output width is zero.
        """
        if (self.high < other.low) or (self.low > other.high):
            return null_band()
        low = max(self.low, other.low)
        high = min(self.high, other.high)
        return Band(0.5 * (low + high), high - low)

    # same syntax as Python set objects
    __and__ = intersection

    @property
    def isvalid(self):
        "True when width is greater than zero."
        return self.width > 0.0


@dataclass(frozen=True)
class PolChannel:
    """
    A polarimetric radar channel.

    Attributes
    ----------
    freq_id : str in {'A', 'B'}
        Sub-band identifier
    pol : str in {"HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV"}
        Polarization: transmit, then receive.
    band : Band
        RF band
    """
    freq_id: str
    pol: str
    band: Band

    # validate inputs
    def __post_init__(self):
        valid_ids = {"A", "B"}
        if self.freq_id not in valid_ids:
            raise ValueError(
                f"Expected id in {valid_ids} got freq_id={self.id}")
        valid_pols = {"HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV"}
        if self.pol not in valid_pols:
            raise ValueError(
                f"Polarization pol={self.pol} not among {valid_pols}")

    @property
    def txpol(self) -> str:
        "Transmit polarization"
        return self.pol[0]

    @property
    def rxpol(self) -> str:
        "Receive polarization"
        return self.pol[1]

    def intersection(self, other: 'PolChannel') -> 'PolChannel':
        """
        Intersect bands of two channels.  If polarization doesn't match, band
        will be set to null and isvalid property evaluates False.
        Output freq_id always matches self.freq_id (left hand side).
        """
        band = null_band()
        if self.pol == other.pol:
            band = self.band & other.band
        return PolChannel(self.freq_id, self.pol, band)

    __and__ = intersection

    @property
    def isvalid(self):
        "True when bandwidth is greater than zero."
        return self.band.isvalid


class UnsupportedModeIntersection(Exception):
    pass


class PolChannelSet(set):
    """
    Container holding a set of PolChannel objects, which can
    describe all the bands held in a NISAR product.  Intersection of
    PolChannelSet can be used to plan mixed-mode processing scenarios
    involving multiple NISAR products.

    Otherwise acts like Set[PolChannel].
    """
    # Ideally this would be a more generic from_product() method, or maybe we'd
    # add getChannelSet methods to all the product readers.  Gauge interest
    # from others before putting in the extra work.
    @classmethod
    def from_raw(cls, raw: RawBase):
        "Derive PolChannelSet from NISAR L0B (raw data) product."
        l = []
        for freq_id in sorted(raw.polarizations):
            for pol in raw.polarizations[freq_id]:
                fc, fs, K, T = [
                    float(x) for x in raw.getChirpParameters(freq_id, pol[0])]
                l.append(PolChannel(freq_id, pol, Band(fc, abs(K * T))))
        return cls(l)

    @property
    def frequencies(self) -> list[str]:
        """List of frequency sub bands"""
        return sorted(list({chan.freq_id for chan in self}))

    def intersection(self, others: Set[PolChannel], regularize=True) -> 'PolChannelSet':
        """
        Calculate intersection with another PolChannelSet.  Returns empty
        collection if no overlapping data.  All elements are also intersected,
        so the resulting mode is entirely common to both.

        By default the result is also checked/regularized to a valid NISAR mode.
        See `regularized` method for details.
        """
        s = PolChannelSet()
        for mine in self:
            for other in others:
                common = mine & other
                if common.isvalid:
                    s.add(common)
        if regularize:
            return s.regularized()
        return s

    __and__ = intersection

    def regularized(self) -> 'PolChannelSet':
        """
        Enforce constraints needed for a valid NISAR product:
        - Single band per channel, e.g., only one entry for (freq_id=A, pol=HH)
        - All entries for a given freq_id have a common band (e.g., can't have
          20 MHz and 40 MHz both labeled freq_id=A).
        - Re-label freq_id in cases where one band overlaps two others.
        - Invalid bands (bw=0) are excluded.

        Raises UnsupportedModeIntersection if constraints can't be satisfied.
        """
        # group by (freq_id, pol)
        d = defaultdict(list)
        for chan in self:
            key = (chan.freq_id, chan.pol)
            if chan.band.isvalid:
                d[key].append(chan)
        # Modes are designed so that bands don't overlap except in 80 MHz case.
        # Handle that by assigning the upper overlap to freq_id "B", and verify
        # that there's not an unanticipated scenario.
        d1 = d.copy()
        for key, channels in d.items():
            freq, pol = key
            if len(channels) > 1:
                if len(channels) > 2:
                    raise UnsupportedModeIntersection(
                        f"unexpected channels {channels}")
                # Try labeling higher band as "B"
                a, b = channels
                if a.band.center == b.band.center:
                    raise UnsupportedModeIntersection("expected unique center"
                        f" frequencies but got {channels}")
                a, b = (a, b) if a.band.center < b.band.center else (b, a)
                newkey = ("B", pol)
                if newkey in d:
                    if d1[newkey].band != b.band:
                        raise UnsupportedModeIntersection(
                            f"overlapping channels {b} and {d[newkey]}")
                    # else they're the same so do nothing
                else:
                    d1[newkey] = [PolChannel("B", pol, b.band)]
                d1[key] = [a]

        if len(d1) == 0:
            return PolChannelSet()

        # Group entries by (freq_id, band) and ensure that there's only one
        # unique band per freq_id.  If necessary and possible, relabel frequency
        # IDs to avoid failure (e.g., quasi-dual 5+5 mode).
        dfb = defaultdict(list)
        for key, channels in d1.items():
            freq, _ = key
            for chan in channels:
                dfb[(freq, chan.band)].append(chan)

        available_freq_labels = ["A", "B"]
        if len(dfb) > len(available_freq_labels):
            raise UnsupportedModeIntersection("more distinct bands than "
                f"available frequency labels: {list(dfb)}")

        freqs, bands = zip(*dfb)
        if len(freqs) > len(set(freqs)):
            # Need to relabel frequency IDs.  Sort them by center frequency.
            bands = sorted(bands, key = lambda band: band.center)
            band2freq = dict(zip(bands, available_freq_labels))
            dfb = {(band2freq[band], band): channels for (freq, band), channels
                in dfb.items()}

        # Collect channels and ensure consistent freq_id in case relabeled.
        l = []
        for (freq, band), channels in dfb.items():
            for chan in channels:
                assert band == chan.band
                l.append(PolChannel(freq, chan.pol, chan.band))
        return PolChannelSet(l)


def find_overlapping_channel(raw: RawBase, desired: PolChannel) -> PolChannel:
    """
    Search a NISAR L0B file for a channel that overlaps the given one and
    returns it.  Raises ValueError if one cannot be found.
    """
    for chan in PolChannelSet.from_raw(raw):
        if chan.intersection(desired).isvalid:
            return chan
    raise ValueError(f"Raw file does not contain channel intersecting {desired}.")
