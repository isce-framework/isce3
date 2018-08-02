#!/usr/bin/env python3

class commonClass:
    def __init__(self):
        self.nc = 100
        self.nl = 200
        self.nbx = 5
        self.nby = 7
        self.latFilename = 'lat.tif'
        self.lonFilename = 'lon.vrt'
        self.incFilename = 'inc.bin'
        self.mskFilename = 'msk.bin'
        self.vrtFilename = 'topo.vrt'


def test_createGeoTiffFloat():
    from isceextension import pyRaster
    from osgeo import gdal
    import os
    import numpy as np
    gdal.UseExceptions()

    cmn = commonClass()
    if os.path.exists( cmn.latFilename):
        os.remove(cmn.latFilename)

    raster = pyRaster(cmn.latFilename, width=cmn.nc, length=cmn.nl,
                        numBands=1, dtype=gdal.GDT_Float32,
                        driver='GTiff', access=gdal.GA_Update)

    assert( os.path.exists(cmn.latFilename))
    assert( raster.width == cmn.nc )
    assert( raster.length == cmn.nl )
    
    assert( raster.numBands == 1)
    assert( raster.getDatatype() == gdal.GDT_Float32)
    del raster

    data = np.zeros((cmn.nl, cmn.nc))
    data[:,:] = np.arange(cmn.nc)[None,:]

    ds = gdal.Open(cmn.latFilename, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None

def test_createVRTDouble_setGetValue():
    from isceextension import pyRaster
    from osgeo import gdal
    import os
    import numpy as np
    import numpy.testing as npt

    gdal.UseExceptions()

    cmn = commonClass()
    if os.path.exists( cmn.lonFilename):
        os.remove(cmn.lonFilename)

    raster = pyRaster(cmn.lonFilename, width=cmn.nc, length=cmn.nl,
                        numBands=1, dtype=gdal.GDT_Float64,
                        driver='VRT', access=gdal.GA_Update)

    assert( os.path.exists(cmn.lonFilename))
    assert( raster.getDatatype() == gdal.GDT_Float64)
    del raster

    data = np.zeros((cmn.nl, cmn.nc))
    data[:,:] = np.arange(cmn.nl)[:,None]

    ##Open and populate
    ds = gdal.Open(cmn.lonFilename, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(data)
    arr = ds.GetRasterBand(1).ReadAsArray()
    npt.assert_array_equal(data, arr, err_msg='RW in Update mode')
    ds = None

    ##Read array
    ds = gdal.Open(cmn.lonFilename, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    npt.assert_array_equal(data, arr, err_msg='Readonly mode')


def test_createTwoBandEnvi():
    from isceextension import pyRaster
    from osgeo import gdal
    import os
    import numpy as np
    gdal.UseExceptions()

    cmn = commonClass()
    if os.path.exists( cmn.incFilename):
        os.remove(cmn.incFilename)

    raster = pyRaster(cmn.incFilename, width=cmn.nc, length=cmn.nl,
                        numBands=2, dtype=gdal.GDT_Int16,
                        driver='ENVI', access=gdal.GA_Update)

    assert( os.path.exists(cmn.incFilename))
    assert( raster.width == cmn.nc )
    assert( raster.length == cmn.nl )
    
    assert( raster.numBands == 2)
    assert( raster.getDatatype() == gdal.GDT_Int16)
    del raster

def test_createMultiBandVRT():
    from isceextension import pyRaster
    from osgeo import gdal
    import os
    gdal.UseExceptions()

    cmn = commonClass()
    lat = pyRaster(cmn.latFilename)
    lon = pyRaster(cmn.lonFilename)
    inc = pyRaster(cmn.incFilename)

    if os.path.exists( cmn.vrtFilename):
        os.remove(cmn.vrtFilename)

    vrt = pyRaster(cmn.vrtFilename, collection=[lat,lon,inc])

    assert( vrt.width == cmn.nc)
    assert( vrt.length == cmn.nl)
    assert( vrt.numBands == 4)
    assert( vrt.getDatatype(1) == gdal.GDT_Float32)
    assert( vrt.getDatatype(2) == gdal.GDT_Float64)
    assert( vrt.getDatatype(3) == gdal.GDT_Int16)
    assert( vrt.getDatatype(4) == gdal.GDT_Int16)

    vrt = None

def test_createNumpyDataset():
    import numpy as np
    from isceextension import pyRaster
    from osgeo import gdal, gdal_array
    import os
    gdal.UseExceptions()

    ny, nx = 200, 100
    data = np.random.randn(ny, nx).astype(np.float32)
    
    dset = gdal_array.OpenArray(data)
    raster = pyRaster('', dataset=dset)

    assert(raster.width == nx)
    assert(raster.length == ny)
    assert(raster.getDatatype() == 6)

    dset = None
    del raster


