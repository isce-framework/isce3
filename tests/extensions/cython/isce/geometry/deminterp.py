# -*- coding: utf-8 -*-

import pytest

def createTiffDEM(epsg, filename):
    from osgeo import gdal, osr
    import numpy as np

    if gdal.VersionInfo()[0] == '3':
        latIndex = 0
    else:
        latIndex = 1

    nY = 120   #Number of lines
    nX = 100   #Number of samples
    delta = 1000. 

    if ((epsg > 32600) and (epsg < 32661)):
        X0 = 500000. - delta * nX//2
        Y0 = 4700000. + delta * nY//2
    elif ((epsg > 32700) and (epsg < 32761)):
        X0 = 500000. - delta * nX//2
        Y0 = 5500000. + delta * nY//2
    elif ((epsg == 3031) or (epsg == 3413)):
        X0 = 0. - delta * nX//2
        Y0 = 0. + delta * nY/2
    else:
        raise ValueError('Unknown EPSG')

    ###Create transformer
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(epsg)

    ###Create latlon transformer
    latlon = osr.SpatialReference()
    latlon.ImportFromEPSG(4326)

    ###Transformer 
    trans = osr.CoordinateTransformation(proj, latlon)

    mat = np.zeros((nY,nX)).astype(np.float32)
    y = (np.arange(nY) + 0.5) * -delta + Y0
    x = (np.arange(nX) + 0.5) * delta + X0

    ###Use 100 * Latitude as data as its a continuous field
    for ii in range(nY):
        for jj in range(nX):
            res = trans.TransformPoint(x[jj], y[ii], 0.)
            mat[ii,jj] = res[latIndex] * 100. 
    
    ###Get geotiff driver
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(filename, nX, nY, 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(mat)
    ds.SetGeoTransform([X0, delta, 0., Y0, 0., -delta])
    ds.SetProjection( proj.ExportToWkt())
    ds = None

    return trans

#Unit test projection systems
testdata = [32601, 32613, 32625, 32637, 32650, 32660,
            32701, 32713, 32725, 32727, 32748, 32760,
            3031, 3413]

def idfn(val):
    return "epsg_{0}".format(val)

@pytest.mark.parametrize("code", testdata, ids=idfn)
def test_deminterp(code):
    '''
    Unit test for dem interpolator.
    '''
    import numpy as np
    import numpy.testing as npt
    import os 
    from osgeo import gdal
    import isce3.extensions.isceextension as isceextension

    if gdal.VersionInfo()[0] == '3':
        latIndex = 0
    else:
        latIndex = 1


    tifffile = "epsg{0}.tif".format(code)
    if os.path.exists(tifffile):
        os.remove(tifffile)

    ###Create the file
    trans = createTiffDEM(code, tifffile)

    ###Create the raster
    raster = isceextension.pyRaster(tifffile)

    ####Create interpolator
    intp = isceextension.pyDEMInterpolator()

    ###Get geotransform
    geotrans = raster.GeoTransform
    width = raster.width
    lgth = raster.length
    minX = geotrans[0]
    maxX = minX + width * geotrans[1]
    maxY = geotrans[3]
    minY = maxY + geotrans[5] * lgth

    ###Load the DEM
    intp.loadDEM(raster)

    
    ### Pick 25 points in random
    Xs = geotrans[0] + np.linspace(5, width-6,5) * geotrans[1]
    Ys = geotrans[3] + np.linspace(5, width-6,5) * geotrans[5]


    ###Test values first
    for ii in range(5):
        for jj in range(5):
            res = trans.TransformPoint(Xs[ii], Ys[jj], 0.)
            val = 100.0 * res[latIndex]

            hxy = intp.interpolateXY(Xs[ii], Ys[jj])
            npt.assert_almost_equal(hxy, val, decimal=3)


            hll = intp.interpolateLonLat(np.radians(res[1-latIndex]), np.radians(res[latIndex]))
            npt.assert_almost_equal(hll, val, decimal=3)

    if os.path.exists(tifffile):
        os.remove(tifffile) 
