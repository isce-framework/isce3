#!/usr/bin/env python3

import numpy
from osgeo import gdal
gdal.UseExceptions()

import h5py

def testh5pyReadOnly():
    import isce3
    import os

    #Create and write to file
    fid = h5py.File('test.h5', 'w')
    ds = fid.create_dataset('data', shape=(50,70), dtype=numpy.float32)
    ds[:] = numpy.arange(50*70).reshape((50,70))


    img = isce3.io.raster(h5=ds)
    assert(img.width == 70)
    assert(img.length == 50)
    assert(img.numBands == 1)
    assert(img.getDatatype() == gdal.GDT_Float32)
    assert(img.isReadOnly == True)
    del img

    fid.close()
    os.remove('test.h5')

    return

def testh5pyUpdate():
    import isce3
    import os

    #Create and write to file
    fid = h5py.File('test.h5', 'w')
    ds = fid.create_dataset('data', shape=(50,70), dtype=numpy.float32)
    ds[:] = numpy.arange(50*70).reshape((50,70))


    img = isce3.io.raster(h5=ds, access=gdal.GA_Update )
    assert(img.width == 70)
    assert(img.length == 50)
    assert(img.numBands == 1)
    assert(img.getDatatype() == gdal.GDT_Float32)
    assert(img.isReadOnly == False)
    del img

    fid.close()
    os.remove('test.h5')

    return

if __name__ == '__main__':
    testh5pyReadOnly()
    testh5pyUpdate()
