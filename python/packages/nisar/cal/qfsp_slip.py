from enum import Flag, unique
import numpy as np
from typing import Sequence

from nisar.products.readers.instrument import InstrumentParser


@unique
class AnomalyCode(Flag):
    """
    NISAR data anomaly codes (bit flags)
    """
    NO_ANOMALY = 0
    SLIP_QFSP_H0 = 1 << 0
    SLIP_QFSP_H1 = 1 << 1  # only one seen in LSAR as of 2026-03-27
    SLIP_QFSP_H2 = 1 << 2
    SLIP_QFSP_V0 = 1 << 3
    SLIP_QFSP_V1 = 1 << 4
    SLIP_QFSP_V2 = 1 << 5
    SLIP_SSAR = 1 << 6
    RESERVED = 1 << 7


def abs2(z):
    return z.real**2 + z.imag**2


# Type hints: (start, end) of elevation (EL) angle interval
ELAngleInterval = tuple[float, float]
# There can be multiple intervals associated with each anomaly.
Boundaries = dict[AnomalyCode, Sequence[ELAngleInterval]]

def get_qfsp_mask_boundaries(anomaly_code: AnomalyCode | int,
                             int_cal: InstrumentParser) -> Boundaries:
    """
    Determine EL angle intervals associated with NISAR anomaly codes.

    Parameters
    ----------
    anomaly_code : AnomalyCode | int
        Bitwise OR of anomaly codes of interest.
    int_cal : InstrumentParser
        NISAR LSAR INT_CAL file containing the angle-to-coefficient (AC) tables.

    Returns
    -------
    boundaries : Boundaries
        Dictionary with a list of EL angle (start, end) intervals for each
        nonzero bit in `anomaly_code`.  Angles are given in radians.
    """
    anomaly_code = AnomalyCode(anomaly_code)

    peak_angles = dict()
    for rxpol in ("H", "V"):
        coeff = int_cal.get_angle2coef(rxpol)
        angles = int_cal.el_angles_ac(rxpol)
        peak_indices = np.argmax(abs2(coeff), axis=1)
        # TODO Could interpolate to find peak.
        peak_angles[rxpol] = np.array([angles[i, j] for (i, j) in
            enumerate(peak_indices)])

    # (start, end) EL angles between beam x and y peaks
    overlap_h_4_5 = peak_angles["H"][3:5]
    overlap_h_8_9 = peak_angles["H"][7:9]
    overlap_v_4_5 = peak_angles["V"][3:5]
    overlap_v_8_9 = peak_angles["V"][7:9]

    boundaries = dict()
    if anomaly_code & AnomalyCode.SLIP_QFSP_H0:
        boundaries[AnomalyCode.SLIP_QFSP_H0] = (overlap_h_4_5,)
    if anomaly_code & AnomalyCode.SLIP_QFSP_H1:
        boundaries[AnomalyCode.SLIP_QFSP_H1] = (overlap_h_4_5, overlap_h_8_9)
    if anomaly_code & AnomalyCode.SLIP_QFSP_H2:
        boundaries[AnomalyCode.SLIP_QFSP_H2] = (overlap_h_8_9,)
    if anomaly_code & AnomalyCode.SLIP_QFSP_V0:
        boundaries[AnomalyCode.SLIP_QFSP_V0] = (overlap_v_4_5,)
    if anomaly_code & AnomalyCode.SLIP_QFSP_V1:
        boundaries[AnomalyCode.SLIP_QFSP_V1] = (overlap_v_4_5, overlap_v_8_9)
    if anomaly_code & AnomalyCode.SLIP_QFSP_V2:
        boundaries[AnomalyCode.SLIP_QFSP_V2] = (overlap_v_8_9,)

    return boundaries
