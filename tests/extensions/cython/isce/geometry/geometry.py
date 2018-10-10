#!/usr/bin/env python3

import numpy as np
import isceextension

def test_geo2rdr():
    """
    Test single call to geo2rdr.
    """
    import datetime
    import gdal

    # Open the HDF5 SLC product for the master scene
    h5 = isceextension.pyIH5File('../../../../lib/isce/data/envisat.h5')

    # Open groups necessary for ISCE objects
    idGroup = h5.openGroup('/science/metadata/identification')
    orbGroup = h5.openGroup('/science/metadata/orbit')
    dopGroup = h5.openGroup('/science/metadata/instrument_data/doppler_centroid')
    modeGroup = h5.openGroup('/science/complex_imagery')

    # Make ISCE objects
    ellps = isceextension.pyEllipsoid()
    orbit = isceextension.pyOrbit()
    doppler = isceextension.pyPoly2d()
    mode = isceextension.pyImageMode()
    
    # Configure objects using metadata 
    isceextension.deserialize(idGroup, ellps)
    isceextension.deserialize(orbGroup, orbit)
    isceextension.deserialize(dopGroup, doppler)
    isceextension.deserialize(modeGroup, mode)
    
    # Call geo2rdr
    llh = [np.radians(-115.6), np.radians(35.1), 55.0]
    aztime, slantrange = isceextension.py_geo2rdr(llh, ellps, orbit, doppler, mode,
                                                  threshold=1.0e-8)

    # Test it
    np.testing.assert_almost_equal(aztime, 1046282126.487007976, decimal=9)
    np.testing.assert_almost_equal(slantrange, 831834.3551143121, decimal=6)

    # Repeat with zero doppler
    zero_doppler = isceextension.pyPoly2d()
    aztime, slantrange = isceextension.py_geo2rdr(llh, ellps, orbit, zero_doppler, mode,
                                                  threshold=1.0e-8)

    # Test it
    np.testing.assert_almost_equal(aztime, 1046282126.613449931, decimal=9)
    np.testing.assert_almost_equal(slantrange, 831833.869159697, decimal=6)


# end of file
