#!/usr/bin/env python3
import iscetest
from pathlib import Path
import numpy as np

from nisar.cal.qfsp_slip import AnomalyCode, get_qfsp_mask_boundaries
from nisar.products.readers.instrument import InstrumentParser


def test_qfsp_boundaries():
    fn = Path(iscetest.data) / "bf" / "REE_INSTRUMENT_TABLE.h5"
    int_cal = InstrumentParser(fn)

    # Call with integer code (don't require enum type).
    boundaries = get_qfsp_mask_boundaries(2, int_cal)
    print(boundaries)

    # We only set a single bit, so we should only have one key.
    assert len(boundaries) == 1

    # There should be two bad regions for middle qFSP slip.
    el_intervals = boundaries[AnomalyCode.SLIP_QFSP_H1]
    assert len(el_intervals) == 2

    # There should be two numbers (start, end) for each interval.
    for start, end in el_intervals:
        # They should be sorted.
        assert end >= start
        # NISAR beams are around one degree wide and spacing isn't much bigger,
        # each region should be roughly that size.
        assert (end - start) < np.deg2rad(1.5)
