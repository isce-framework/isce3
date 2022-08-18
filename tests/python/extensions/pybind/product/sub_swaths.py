#!/usr/bin/env python3

import numpy.testing as npt

import isce3.ext.isce3 as isce3
import iscetest

def test_sub_swaths():

    # Create SubSwaths object
    swath = isce3.product.Swath(iscetest.data + 'Greenland.h5', 'A')
    sub_swaths = swath.sub_swaths()

    # Check its values
    assert sub_swaths.num_sub_swaths == 1

    # Get valid_samples_array_1 array
    valid_samples_array_1 = sub_swaths.get_valid_samples_array(1)

    # Check number of lines and samples
    assert sub_swaths.length == swath.lines
    assert sub_swaths.width == swath.samples

    # Check length (number of az. lines)
    assert valid_samples_array_1.shape[0] == swath.lines

    # Check width == 2 representing start and end range for each az. line
    assert valid_samples_array_1.shape[1] == 2

    # Invalid az and rg indexes should return 0
    assert sub_swaths.get_sample_sub_swath(-1, -1) == 0
    assert sub_swaths.get_sample_sub_swath(-1, 0) == 0
    assert sub_swaths.get_sample_sub_swath(0, -1) == 0
    assert sub_swaths.get_sample_sub_swath(swath.lines, 0) == 0
    assert sub_swaths.get_sample_sub_swath(swath.lines, 1) == 0
    assert sub_swaths.get_sample_sub_swath(swath.lines + 1, 0) == 0
    assert sub_swaths.get_sample_sub_swath(swath.lines + 1, 1) == 0

    # Use valid values from valid_samples_array_1
    # swath.lines//1000 = 16838//1000 = 16
    r0 = 0
    rf = swath.samples
    for i in range(0, swath.lines, 1000):

        r0 = valid_samples_array_1[i, 0]
        rf = valid_samples_array_1[i, 1]

        assert sub_swaths.get_sample_sub_swath(i, -1) == 0
        assert sub_swaths.get_sample_sub_swath(i, r0 - 1) == 0
        assert sub_swaths.get_sample_sub_swath(i, r0) == 1
        assert sub_swaths.get_sample_sub_swath(i, (r0+rf)//2) == 1
        assert sub_swaths.get_sample_sub_swath(i, rf - 1) == 1
        # half-open interval: sample at rf is invalid
        assert sub_swaths.get_sample_sub_swath(i, rf) == 0
        assert sub_swaths.get_sample_sub_swath(i, rf + 1) == 0

    # Get last valid azimuth line and use r0 and rf from last solution above
    i = swath.lines - 1
    assert sub_swaths.get_sample_sub_swath(i, -1) == 0
    assert sub_swaths.get_sample_sub_swath(i, r0 - 1) == 0
    assert sub_swaths.get_sample_sub_swath(i, r0) == 1
    assert sub_swaths.get_sample_sub_swath(i, (r0+rf)//2 ) == 1
    assert sub_swaths.get_sample_sub_swath(i, rf - 1 ) == 1
    # half-open interval: sample at rf is invalid
    assert sub_swaths.get_sample_sub_swath(i, rf) == 0
    assert sub_swaths.get_sample_sub_swath(i, rf + 1) == 0

    # Get next line azimuth line (invalid) and use r0 and rf from
    # solution above
    i = swath.lines
    assert sub_swaths.get_sample_sub_swath(i, -1) == 0
    assert sub_swaths.get_sample_sub_swath(i, r0 - 1) == 0
    assert sub_swaths.get_sample_sub_swath(i, r0) == 0
    assert sub_swaths.get_sample_sub_swath(i, (r0+rf)//2 ) == 0
    assert sub_swaths.get_sample_sub_swath(i, rf - 1 ) == 0
    assert sub_swaths.get_sample_sub_swath(i, rf) == 0
    assert sub_swaths.get_sample_sub_swath(i, rf + 1) == 0
