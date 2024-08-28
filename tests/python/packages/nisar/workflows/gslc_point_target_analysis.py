import iscetest
import os
import numpy.testing as npt
from nisar.workflows.gslc_point_target_analysis import (
    analyze_gslc_point_targets_csv,
    analyze_gslc_point_target_llh,
)
import pytest


@pytest.mark.parametrize("test_args", [
    (
        analyze_gslc_point_target_llh,
        dict(cr_llh=[-54.579586258, 3.177088785, 0.0], in_rads=False)
    ),
    (
        analyze_gslc_point_targets_csv,
        dict(corner_reflector_csv=os.path.join(iscetest.data, "REE_CR_INFO_out17.csv"))
    ),
])
@pytest.mark.parametrize("kwargs", [
    dict(upsample_factor=16),
    dict(peak_find_domain='frequency'),
    dict(cuts=True),
])
def test_point_target_analysis(test_args, kwargs):
    """
    Test the process that projects input point target lon/lat/height to the product
    geo-grid coordinates and computes the x/y offsets of predicted point target
    location w.r.t the observed point target location within the L2 GSLC.
    """
    gslc_name = os.path.join(iscetest.data, "REE_GSLC_out17_unflattened.h5")
    freq_group = 'A'
    polarization = 'HH'

    func, cr_kwargs = test_args

    results = func(
        gslc_filename=gslc_name,
        output_file=None,
        freq=freq_group,
        pol=polarization,
        **cr_kwargs,
        **kwargs
    )

    performance_dict = results if func == analyze_gslc_point_target_llh else results[0]

    x_offset = performance_dict['x']['offset']
    y_offset = performance_dict['y']['offset']

    # Compare slant range offset and azimuth offset against default values.
    # The GSLC offsets tend to be off by about half a pixel 
    npt.assert_(abs(x_offset) < 0.6,
        f'X offset {x_offset} is larger than expected.')
    npt.assert_(abs(y_offset) < 0.6,
        f'Y offset {y_offset} is larger than expected.')
