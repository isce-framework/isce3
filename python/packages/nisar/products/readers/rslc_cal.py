from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from isce3.core import DateTime
import logging
from numpy import angle, deg2rad, rad2deg, exp, pi
from ruamel.yaml import YAML
from typing import List, Optional, Tuple, Union


log = logging.getLogger("rslc_cal")


@dataclass
class PolChannelParams:
    """
    Polarization-specific calibration parameters

    Attributes
    ----------
    delay : float
        Range delay in meters. It supports both negative (lead) and positive (lag) value.
    scale : complex
        Complex amplitude correction which can be used to implement absolute
        and polarimetric calibration [1]_
    scale_slope : float
        Slope of magnitude correction wrt EL (per radian)

    Notes
    -----
    .. [1] A. G. Fore et al., "UAVSAR Polarimetric Calibration," in IEEE Trans.
       Geoscience and Remote Sensing, vol. 53, no. 6, pp. 3481-3491, June 2015,
       doi: 10.1109/TGRS.2014.2377637.
    """
    delay: float = 0.0
    scale: complex = 1+0j
    scale_slope: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> PolChannelParams:
        """
        Construct PolChannelParams object from a dictionary

        Parameters
        ----------
        d : dict
            Dictionary containing keys {"Differential Delay (m)",
            "Differential Phase (degrees)", "Radiometric Scale Factor",
            "Radiometric Scale Factor Slope (/degree)"}
        """
        d = d.copy()
        delay = d.pop("Differential Delay (m)", cls.delay)
        default_mag = abs(cls.scale)
        default_phs_deg = angle(cls.scale, deg=True)
        phs = deg2rad(d.pop("Differential Phase (degrees)", default_phs_deg))
        mag = d.pop("Radiometric Scale Factor", default_mag)
        scale = mag * exp(1j * phs)
        slope = d.pop("Radiometric Scale Factor Slope (/degree)",
                      cls.scale_slope * pi/180) * 180/pi
        for key in d:
            log.warning(f"did not parse calibration sub {key = }")
        return cls(delay, scale, slope)


def bandwidth_to_key(bandwidth: float) -> str:
    """
    Convert radar bandwidth to a dictionary key.

    Parameters
    ----------
    bandwidth : float
        Radar bandwidth in Hz.

    Returns
    -------
    key : str
        Dictionary key for bandwidth-specific overrides.  Follows the pattern
        "Override X MHz" where X is one of the nominal NISAR bandwidths in
        {5, 20, 40, 77}.
    """
    if (bandwidth < 0.0) or (bandwidth > 85e6):
        raise ValueError(f"Expected 0 < B <= 80e6 Hz got B={bandwidth}")
    # Use decision points halfway between nominal bandwidth values to handle
    # finite precision.
    B = 5
    if bandwidth > 10e6:
        B = 20
    if bandwidth > 30e6:
        B = 40
    if bandwidth > 60e6:
        B = 77
    return f"Override {B} MHz"


@dataclass
class RslcCalibration:
    """
    Parameters needed to produce a calibrated NISAR RSLC product.
    Note that compact-pol (circular transmit) is not currently supported.

    Attributes
    ----------
    hh, hv, vh, vv : PolChannelParams
        Channel-specific calibration parameters
    notes : str
        Any free-form notes provided in the calibration file.
    generated_date : datetime
        Date when cal parameters were generated
    valid_after_date : datetime
        Date after which these parameters are valid
    valid_before_date : datetime or None
        End date where the parameters become invalid (if known)
    common_delay : float
        Range delay to apply to all channels (in meters)
    reference_range : float
        Range used to normalize range fading correction (in meters)
    """
    # Workaround for mutable members, see
    # https://docs.python.org/3/library/dataclasses.html#mutable-default-values
    hh: PolChannelParams = field(default_factory=PolChannelParams)
    hv: PolChannelParams = field(default_factory=PolChannelParams)
    vh: PolChannelParams = field(default_factory=PolChannelParams)
    vv: PolChannelParams = field(default_factory=PolChannelParams)
    notes: str = ""
    generated_date: datetime = datetime.now(timezone.utc)
    valid_after_date: datetime = datetime(1978, 6, 27)  # before Seasat launch
    valid_before_date: Optional[datetime] = None
    common_delay: float = 0.0  # m
    reference_range: float = 900.0e3  # m

    @classmethod
    def from_dict(cls, d: dict, bandwidth: Optional[float] = None) -> RslcCalibration:
        """
        Construct RslcCalibration from dictionary

        Parameters
        ----------
        d : dict
            Dictionary containing keys found in share/nisar/schemas/rslc_calibration.yaml
        bandwidth : float or None, optional
            Bandwidth of interest (in Hz) in case the calibration has any
            bandwidth-specific parameter overrides.
        """
        # Copy so we can consume keys and check for any that weren't parsed.
        d = d.copy()

        # get defaults and override with bandwidth-specific values
        all_cal = d.pop("Calibration", {})
        cal = all_cal.pop("Default", {})
        if bandwidth is not None:
            key = bandwidth_to_key(bandwidth)
            if key in all_cal:
                cal.update(all_cal[key])
            else:
                log.info(f"No '{key}' key found, using 'Default' section")
        for key in all_cal:
            if not key.startswith("Override"):
                log.warning(f"Did not parse calibration {key = }")

        channels = defaultdict(PolChannelParams)
        for key in ("HH", "HV", "VH", "VV"):
            if key in cal:
                channels[key] = PolChannelParams.from_dict(cal.pop(key))

        common_delay = cal.pop("Common Delay (m)", cls.common_delay)

        for key in cal:
            log.warning(f"did not parse calibration sub {key = }")

        notes = d.pop("Notes", cls.notes)
        # NOTE YAML parser already converts datetime objects
        generated_date = d.pop("Date Generated", cls.generated_date)
        valid_after_date = d.pop("Valid for Data Acquired After Date",
                                 cls.valid_after_date)
        valid_before_date = d.pop("Valid for Data Acquired Before Date",
                                  cls.valid_before_date)
        reference_range = d.pop("Reference Range (m)", cls.reference_range)

        for key in d:
            log.warning(f"did not parse calibration {key = }")

        return RslcCalibration(hh=channels["HH"], hv=channels["HV"],
                               vh=channels["VH"], vv=channels["VV"],
                               common_delay=common_delay,
                               notes=notes, generated_date=generated_date,
                               valid_after_date=valid_after_date,
                               valid_before_date=valid_before_date,
                               reference_range=reference_range)
                            
    def __post_init__(self):
        if self.reference_range <= 0.0:
            raise ValueError(
                f"Expected reference range > 0 but got {reference_range} m.")



def parse_rslc_calibration(filename: str, bandwidth: Optional[float] = None) -> RslcCalibration:
    """
    Construct RslcCalibration from YAML file

    Parameters
    ----------
    filename : str
        YAML file conforming to schema in share/nisar/schemas/rslc_calibration.yaml
    bandwidth : float
        Bandwidth of interest (in Hz) in case the calibration has any
        bandwidth-specific parameter overrides.

    Returns
    -------
    cal : RslcCalibration
        RSLC calibration parameters
    """
    parser = YAML(typ="safe")
    with open(filename) as f:
        cfg = parser.load(f)
    return RslcCalibration.from_dict(cfg, bandwidth=bandwidth)


def get_scale_and_delay(cal: RslcCalibration, pol: str) -> Tuple[complex, float]:
    """
    Get scale factor and delay for a particular polarization.

    Parameters
    ----------
    cal : RslcCalibration
        Data class containing NISAR calibration data for the desired band.
    pol : str in {"HH", "HV", "VH", "VV"}
        Polarization of interest

    Returns
    -------
    scale : complex
        Multiplicative scale factor (linear amplitude) to radiometrically
        calibrate the data.
    delay : float
        Total delay (common delay + differential delay) in meters for the
        requested polarimetric channel.
    """
    pol = pol.lower()
    if pol not in ("hh", "hv", "vh", "vv"):
        raise NotImplementedError("Only linear polarizations are supported, "
            f"requested polarization={pol.upper()}")
    params = getattr(cal, pol)
    if abs(params.scale_slope) > 0.0:
        raise NotImplementedError(
            "Range-varying amplitude correction is not implemented")

    delay = cal.common_delay + params.delay
    scale = params.scale
    return scale, delay


def check_cal_validity_dates(cal: RslcCalibration,
                             observation_start: Union[DateTime, datetime],
                             observation_end: Union[DateTime, datetime] = None):
    """
    Raise an exception if radar time is not between the validity dates in the
    calibration file.

    Parameters
    ----------
    cal : RslcCalibration
        NISAR calibration parameter object
    observation_start : datetime.datetime or isce3.core.DateTime
        Observation start time of radar data to be calibrated
    observation_end : datetime.datetime or isce3.core.DateTime, optional
        Observation end time of radar data to be calibrated
        If not provided observation_start is used in its place.
    """
    t0 = t1 = DateTime(observation_start)
    if observation_end is not None:
        t1 = DateTime(observation_end)
    if t1 < t0:
        raise ValueError("Expected end time >= start time")
    if not (t0 >= DateTime(cal.valid_after_date)):
        raise ValueError(
            f"Calibration file validity start date {cal.valid_after_date} "
            f"occurs after radar observation start time {t0}")
    if ((cal.valid_before_date is not None)
            and not (t1 <= DateTime(cal.valid_before_date))):
        raise ValueError(
            f"Calibration file validity end date {cal.valid_before_date} "
            f"occurs before radar observation end time {t1}")

