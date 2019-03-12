#!/usr/bin/env python3

import numpy as np
import isce3.extensions.isceextension as isceextension

def test_geo2rdr():
    """
    Test single call to geo2rdr.
    """
    import datetime
    from osgeo import gdal

    # Open the HDF5 SLC product for the master scene
    h5 = isceextension.pyIH5File('../../../../lib/isce/data/envisat.h5')

    # Create product
    product = isceextension.pyProduct(h5)

    # Make ISCE objects
    ellps = isceextension.pyEllipsoid()
    orbit = product.metadata.orbit
    doppler = product.metadata.procInfo.dopplerCentroid(freq='A')
    wvl = product.swathA.processedWavelength
        
    # Call geo2rdr
    llh = [np.radians(-115.72466801139711), np.radians(34.65846532785868), 1772.0]
    aztime, slantrange = isceextension.py_geo2rdr(llh, ellps, orbit, doppler, wvl,
                                                  threshold=1.0e-10, dR=10.0)

    # Test it
    np.testing.assert_almost_equal(aztime, 237333.993088889, decimal=9)
    np.testing.assert_almost_equal(slantrange, 830450.1859454865, decimal=6)

    # Repeat with zero doppler
    zero_doppler = isceextension.pyLUT2d()
    aztime, slantrange = isceextension.py_geo2rdr(llh, ellps, orbit, zero_doppler, wvl,
                                                  threshold=1.0e-10, dR=10.0)

    # Test it
    np.testing.assert_almost_equal(aztime, 237334.1228937040, decimal=9)
    np.testing.assert_almost_equal(slantrange, 830449.6727720449, decimal=6)


# end of file
