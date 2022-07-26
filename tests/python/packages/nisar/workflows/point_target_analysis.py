import iscetest
import numpy as np
import os
import numpy.testing as npt
from nisar.workflows.point_target_analysis import slc_pt_performance


def get_test_file():
    rslc_name = os.path.join(iscetest.data, "rslc_pt.h5")
    
    return rslc_name

def test_cr_geo2rdr():
    """ Test process that converts input point target lon/lat/ht to radar
        coordinates and computes the range/azimuth offsets of predicted
        point target location w.r.t observed point target location within 
        the L1 RSLC
    """

    rslc_name = get_test_file()
    freq_group = 'A'
    polarization = 'HH'
    cr_llh = np.array([-54.58, 3.177, 0])
    fs_bw_ratio = 1.2
    num_sidelobes = 10
    predict_null = True
    nov = 32
    chipsize = 64
    plots = False
    cuts = False
    window_type = 'rect'
    window_parameter = 0

    performance_dict = slc_pt_performance(
        rslc_name,
        freq_group,
        polarization,
        cr_llh,
        fs_bw_ratio,
        num_sidelobes,
        predict_null,
        nov,
        chipsize,
        plots,
        cuts,
        window_type,
        window_parameter,
    )

    slant_range_offset = performance_dict['range']['offset']
    azimuth_offset = performance_dict['azimuth']['offset']

    #Compare slant range offset and azimuth offset against default values
    npt.assert_equal(slant_range_offset, 4.96875, 'Slant range bin offset is larger than expected.')
    npt.assert_equal(azimuth_offset, 0, 'Azimuth bin offset is larger than expected.')
