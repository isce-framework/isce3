import iscetest
import numpy as np
import os
import numpy.testing as npt
from nisar.workflows.point_target_analysis import slc_pt_performance
import pytest


@pytest.mark.parametrize("kwargs", [
    dict(nov=16),
    dict(shift_domain='frequency'),
    dict(predict_null=True, window_type='kaiser', window_parameter=1.6),
    dict(cuts=True),
])
def test_point_target_analysis(kwargs):
    """ Test process that converts input point target lon/lat/ht to radar
        coordinates and computes the range/azimuth offsets of predicted
        point target location w.r.t observed point target location within
        the L1 RSLC
    """
    rslc_name = os.path.join(iscetest.data, "REE_RSLC_out17.h5")
    freq_group = 'A'
    polarization = 'HH'
    cr_llh = [-54.579586258, 3.177088785, 0.0]  # lon, lat, hgt in (deg, deg, m)

    performance_dict = slc_pt_performance(
        rslc_name,
        freq_group,
        polarization,
        cr_llh,
        **kwargs
    )

    slant_range_offset = performance_dict['range']['offset']
    azimuth_offset = performance_dict['azimuth']['offset']

    #Compare slant range offset and azimuth offset against default values
    npt.assert_(abs(slant_range_offset) < 0.1,
        f'Slant range bin offset {slant_range_offset} is larger than expected.')
    npt.assert_(abs(azimuth_offset) < 0.1,
        f'Azimuth bin offset {azimuth_offset} is larger than expected.')
