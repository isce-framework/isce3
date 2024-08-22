import iscetest
import os
import numpy.testing as npt
from nisar.workflows.gslc_point_target_analysis import analyze_gslc_point_target_llh
import pytest


@pytest.mark.parametrize("kwargs", [
    dict(upsample_factor=16),
    dict(peak_find_domain='frequency'),
    dict(cuts=True),
])
def test_point_target_analysis(kwargs):
    """
    Test the process that projects input point target lon/lat/height to the product
    geo-grid coordinates and computes the x/y offsets of predicted point target
    location w.r.t the observed point target location within the L2 GSLC.
    """
    gslc_name = os.path.join(iscetest.data, "REE_GSLC_out17_unflattened.h5")
    freq_group = 'A'
    polarization = 'HH'
    cr_llh = [-54.579586258, 3.177088785, 0.0]  # lon, lat, height in (deg, deg, m)

    performance_dict = analyze_gslc_point_target_llh(
        gslc_filename=gslc_name,
        output_file=None,
        freq=freq_group,
        pol=polarization,
        cr_llh=cr_llh,
        in_rads=False,
        **kwargs
    )

    x_offset = performance_dict['x']['offset']
    y_offset = performance_dict['y']['offset']

    # Compare slant range offset and azimuth offset against default values.
    # The GSLC offsets tend to be off by about half a pixel 
    npt.assert_(abs(x_offset) < 0.6,
        f'X offset {x_offset} is larger than expected.')
    npt.assert_(abs(y_offset) < 0.6,
        f'Y offset {y_offset} is larger than expected.')
