#!/usr/bin/env python3

import numpy as np
import isceextension

def test_geo2rdr():
    """
    Test single call to geo2rdr.
    """
    import datetime
    import gdal

    # Open the SLC product for the master scene
    dset = gdal.Open('../../../../lib/isce/data/envisat.slc.vrt', gdal.GA_ReadOnly)

    # Get the metadata string
    band = dset.GetRasterBand(1)
    xml = band.GetMetadata('xml:isce')[0]
    dset = None
    
    # Make ISCE objects
    ellps = isceextension.pyEllipsoid()
    orbit = isceextension.pyOrbit()
    doppler = isceextension.pyPoly2d()
    meta = isceextension.pyMetadata()
    
    # Configure objects using metadata 
    ellps.archive(xml)
    orbit.archive(xml)
    meta.archive(xml)
    doppler.archive(xml, 'SkewDoppler')

    # Call geo2rdr
    llh = [np.radians(35.10), np.radians(-115.6), 55.0]
    aztime, slantrange = isceextension.py_geo2rdr(llh, ellps, orbit, doppler, meta,
                                                  threshold=1.0e-8)

    # Test it
    np.testing.assert_almost_equal(aztime, 1046282126.487007976, decimal=9)
    np.testing.assert_almost_equal(slantrange, 831834.3551143121, decimal=6)

    # Repeat with zero doppler
    zero_doppler = isceextension.pyPoly2d()
    aztime, slantrange = isceextension.py_geo2rdr(llh, ellps, orbit, zero_doppler, meta,
                                                  threshold=1.0e-8)

    # Test it
    np.testing.assert_almost_equal(aztime, 1046282126.613449931, decimal=9)
    np.testing.assert_almost_equal(slantrange, 831833.869159697, decimal=6)


# end of file
