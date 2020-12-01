import os

import numpy.testing as npt
from pybind_nisar.workflows import stage_dem

import iscetest


def test_run():
    """
    Run DEM stager
    """
    # Get reference SLC
    slc_file = os.path.join(iscetest.data, 'Greenland.h5')

    # Prepare output directory
    out_file = os.path.join('dem.tif')

    # Return and check S3 VRT Filepath
    vrtFilename = stage_dem.return_dem_filepath(None, ref_slc=slc_file)
    npt.assert_equal(vrtFilename, '/vsis3/nisar-dem/EPSG3413/EPSG3413.vrt')

    # Return and check EPSG
    poly = stage_dem.determine_polygon(slc_file)
    epsg = stage_dem.determine_projection(poly, None)

    npt.assert_equal(epsg, 3413)

    # Download dem
    stage_dem.download_dem(poly, epsg, 5, out_file)
    os.remove(out_file)
