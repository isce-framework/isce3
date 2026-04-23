from enum import Flag, unique
import numpy as np
from typing import Sequence, Union

from isce3.core import LUT2d
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

def get_qfsp_mask_boundaries(anomaly_code: Union[AnomalyCode, int],
                             int_cal: InstrumentParser) -> Boundaries:
    """
    Determine EL angle intervals covering qFSP transition regions of 12-channel
    (three qFSPs) L-band NISAR associated with qFSP sample slip anomaly codes.

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
    if not (peak_angles["H"].size == peak_angles["V"].size == 12):
        nh = peak_angles["H"].size
        nv = peak_angles["V"].size
        raise ValueError(
            f"Expected 12 channels for NISAR L-SAR but got {nh} on H and "
            f"{nv} on V."
        )
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


def write_anomaly_mask(anomaly_code, dataset, t0_axis, r0_axis, tn_lut, rn_lut,
                       el_lut, int_cal):
    """
    Generate anomaly mask and save to HDF5 dataset

    Parameters
    ----------
    anomaly_code : AnomalyCode | int
        Bitwise OR of anomaly codes of interest.
    dataset : h5py.Dataset | array_like
        HDF5 dataset for storing mask.
    t0_axis : isce3.core.Linspace
        Zero-Doppler time axis associated with RSLC image.
    r0_axis : isce3.core.Linspace
        Zero-Doppler range axis associated with RSLC image.
    tn_lut, rn_lut : isce3.core.LUT2d
        Reskew tables providing native Doppler time and range as functions of
        zero-Doppler (time, range).
    el_lut : isce3.core.LUT2d
        Table providing antenna EL angle as a function of native Doppler
        (time, range).
    int_cal : InstrumentParser
        NISAR LSAR INT_CAL file containing the angle-to-coefficient (AC) tables.
    Notes
    -------
    This function generates invalid mask simply for 12-channel L-SAR product
    with qFSP sample slip anomaly.
    """
    nt = t0_axis.size
    nr = r0_axis.size
    if dataset.shape[0] != nt:
        raise ValueError("Mask shape[0] is incompatible with time axis length")
    if dataset.shape[1] != nr:
        raise ValueError("Mask shape[1] is incompatible with range axis length")

    # Full image mask may be too big to fit in memory.  If it's an HDF5 dataset
    # then use chunk size as azimuth block size.
    block_size = nt
    if getattr(dataset, "chunks", None) is not None:
        block_size = dataset.chunks[0]

    def write_block(buf, block):
        dataset[block] = buf
    if hasattr(dataset, "write_direct"):
        write_block = lambda buf, block: dataset.write_direct(buf, dest_sel=block)

    # Figure out the EL intervals we have to mask out.
    boundaries = get_qfsp_mask_boundaries(anomaly_code, int_cal)

    # Before doing anything else, check for no-anomaly since we can return
    # early in that case.
    if len(boundaries) == 0:
        for block_start in range(0, nt, block_size):
            block = slice(block_start, min(block_start + block_size, nt))
            nb = block.stop  - block.start
            buf = np.zeros((nb, dataset.shape[1]), dataset.dtype)
            buf[...] = AnomalyCode(anomaly_code).value
            write_block(buf, np.s_[block, :])
        return

    # Not so lucky.  Now let's generate a mask based on the EL LUT data and the
    # mask intervals.  EL LUT2d is small enough to hold in memory, so mask
    # should be, too.
    el_mask = np.zeros(el_lut.data.shape, dataset.dtype)
    for anomaly_bit, el_intervals in boundaries.items():
        code_mask = np.zeros(el_mask.shape, bool)
        for el_low, el_high in el_intervals:
            code_mask |= (el_lut.data >= el_low) & (el_lut.data <= el_high)
        el_mask[code_mask] |= anomaly_bit.value

    # Construct LUT2d with nearest-neighbor so mask values don't change.
    # Domain of LUT is raw data (native Doppler), so we'll have to reskew it.
    # NOTE This converts the dtype to float64, but that's okay as long as there
    # are fewer than 52 bits in the mask (mantissa of float64).
    mask_lut = LUT2d(el_lut.x_axis, el_lut.y_axis, el_mask, method="nearest",
                     b_error=False)

    for block_start in range(0, nt, block_size):
        block_end = min(block_start + block_size, nt)
        nb = block_end - block_start
        # Allocate each chunk to avoid HDF5 I/O as much as possible.
        mask_chunk = np.full(fill_value=AnomalyCode.NO_ANOMALY.value,
            shape=(nb, nr), dtype=dataset.dtype)
        for i_chunk, i_time in enumerate(range(block_start, block_end)):
            t0 = t0_axis[i_time]
            # Compute native Doppler (time, range) from zero-Doppler ones.
            # Note vectorization along range-axis.
            tn = tn_lut.eval(t0, r0_axis)
            rn = rn_lut.eval(t0, r0_axis)
            mask_chunk[i_chunk, :] = mask_lut.eval(tn, rn)
        # Write to output array / HDF5 dataset.
        write_block(mask_chunk, np.s_[block_start : block_end, :])
