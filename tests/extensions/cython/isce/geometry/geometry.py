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

    # Open groups necessary for ISCE objects
    idGroup = h5.openGroup('/science/metadata/identification')
    orbGroup = h5.openGroup('/science/metadata/orbit')
    dopGroup = h5.openGroup('/science/metadata/instrument_data/doppler_centroid')
    modeGroup = h5.openGroup('/science/complex_imagery')

    # Make ISCE objects
    ellps = isceextension.pyEllipsoid()
    orbit = isceextension.pyOrbit()
    doppler = isceextension.pyLUT1d()
    mode = isceextension.pyImageMode()
    
    # Configure objects using metadata 
    isceextension.deserialize(idGroup, ellps)
    isceextension.deserialize(orbGroup, orbit)
    isceextension.deserialize(dopGroup, doppler, name_coords='r0', name_values='skewdc_values')
    isceextension.deserialize(modeGroup, mode)

    # Update reference epoch for orbit
    ref_epoch = isceextension.pyDateTime()
    ref_epoch.strptime('2003-02-25T00:00:00.000000')
    orbit.updateUTCTimes(ref_epoch)
    
    # Call geo2rdr
    llh = [np.radians(-115.72466801139711), np.radians(34.65846532785868), 1772.0]
    aztime, slantrange = isceextension.py_geo2rdr(llh, ellps, orbit, doppler, mode,
                                                  threshold=1.0e-10, dR=10.0)

    # Test it
    np.testing.assert_almost_equal(aztime, 150933.993088889, decimal=9)
    np.testing.assert_almost_equal(slantrange, 830450.1859454865, decimal=6)

    # Repeat with zero doppler
    zero_doppler = isceextension.pyLUT1d()
    aztime, slantrange = isceextension.py_geo2rdr(llh, ellps, orbit, zero_doppler, mode,
                                                  threshold=1.0e-10, dR=10.0)

    # Test it
    np.testing.assert_almost_equal(aztime, 150934.1228937040, decimal=9)
    np.testing.assert_almost_equal(slantrange, 830449.6727720449, decimal=6)


# end of file
