import os
from iscetest import *

from pybind_nisar.products.readers import SLC
from pybind_nisar.workflows import DEM


def test_run():
    """
    Run DEM stager
    """
    iscetest = isce_test()
    slcFile = os.path.join(iscetest.data,
                           'SanAndreas_metaOnly.h5')
    print(slcFile)
    # Prepare output directory
    outFile = slcFile.replace('SanAndreas_metaOnly.h5',
                              'dem.tif')
    print(outFile)
    # Return VRT Filepath
    productSlc = SLC(hdf5file=slcFile)
    orbit = productSlc.getOrbit()
    radarGrid = productSlc.getRadarGrid()
    doppler = productSlc.getDopplerCentroid(frequency='A')

    vrtFilename = DEM.return_dem_filepath(orbit=orbit, radarGrid=radarGrid,
                                          doppler=doppler)
    print('DEM VRT filepath:', vrtFilename)

    # Download DEM
    ring = DEM.determine_perimeter(slcFile)
    epsg = DEM.determine_projection(ring, None)
    DEM.download_dem(ring, epsg, 1000, outFile, None)


if __name__ == '__main__':
    test_run()
