#cython: language_level=3
#
# Author: Piyush Agram 
# Copyright 2018-2019
#

import numpy as np
cimport numpy as np
from libcpp cimport bool
from DEMInterpolator cimport DEMInterpolator
from Raster cimport Raster
from Interpolator cimport dataInterpMethod
from cython.operator cimport dereference as deref

cdef class pyDEMInterpolator:
    '''
    Cython wrapper for isce::geometry::DEMInterpolator.

    Args:
        refHeight (Optional[float]) : Default reference height for the DEM.
        method (Optional[str]): DEM Interpolation method to use.

    Return:
        None
    '''

    cdef DEMInterpolator *c_deminterp
    cdef bool __owner

    # DEM interpolation methods
    demInterpMethods = {
        'sinc' : dataInterpMethod.SINC_METHOD,
        'bilinear': dataInterpMethod.BILINEAR_METHOD,
        'bicubic': dataInterpMethod.BICUBIC_METHOD,
        'nearest': dataInterpMethod.NEAREST_METHOD,
        'biquintic': dataInterpMethod.BIQUINTIC_METHOD
    }


    def __cinit__(self, height=0., method='biquintic'):
        '''
        Constructor takes in a reference height and an interpolation method.
        '''
        # Create C++ DEMInterpolator pointer
        self.c_deminterp = new DEMInterpolator(height, self.demInterpMethods[method])
        self.__owner = True


    def __dealloc__(self):
        if self.__owner:
            del self.c_deminterp

    def loadDEMSubset(self, pyRaster raster, minX, maxX, minY, maxY):
        '''
        Load specified region from a given DEM.
        '''
        self.c_deminterp.loadDEM(deref(raster.c_raster), minX, maxX, minY, maxY)

    def loadDEM(self, pyRaster raster):
        '''
        Load the entire DEM.
        '''
        self.c_deminterp.loadDEM(deref(raster.c_raster))

    @property
    def xStart(self):
        return self.c_deminterp.xStart()

    @property
    def yStart(self):
        return self.c_deminterp.yStart()

    @property
    def deltaX(self):
        return self.c_deminterp.deltaX()

    @property
    def deltaY(self):
        return self.c_deminterp.deltaY()

    @property
    def midX(self):
        return self.c_deminterp.midX()

    @property
    def midY(self):
        return self.c_deminterp.midY()

    @property
    def haveRaster(self):
        return self.c_deminterp.haveRaster()

    @property
    def refHeight(self):
        return self.c_deminterp.refHeight()

    @refHeight.setter
    def refHeight(self, val):
        self.c_deminterp.refHeight(val)
    
    @property
    def width(self):
        return self.c_deminterp.width()

    @property
    def length(self):
        return self.c_deminterp.length()

    @property
    def epsg(self):
        return self.c_deminterp.epsgCode()

    @property
    def data(self):
        cdef np.float32_t[:,:] view = <np.float32_t[:self.length(),:self.width()]> self.c_deminterp.data()
        return np.asarray(view)

    def atLonLat(self, lon, lat):
        return self.c_deminterp.interpolateLonLat(lon,lat)

    def atXY(self, x, y):
        return self.c_deminterp.interpolateXY(x, y)
