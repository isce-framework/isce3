#cython: language_level=3
#
# Author: Bryan Riel, Joshua Cohen
# Copyright 2017-2018
#

from libcpp cimport bool
import numpy as np
cimport numpy as np
from LUT2d cimport LUT2d

cdef class pyLUT2d:
    '''
    Python wrapper for isce::core::LUT2d

    Args:
        x (ndarray or list): X-Coordinates for LUT
        y (ndarray or list): Y-Coordinates for LUT
        z (ndarray):         Data values for LUT
    '''
    cdef LUT2d[double] * c_lut
    cdef bool __owner

    # DEM interpolation methods
    interpMethods = {
        'sinc': dataInterpMethod.SINC_METHOD,
        'bilinear': dataInterpMethod.BILINEAR_METHOD,
        'bicubic': dataInterpMethod.BICUBIC_METHOD,
        'nearest': dataInterpMethod.NEAREST_METHOD,
        'akima': dataInterpMethod.AKIMA_METHOD,
        'biquintic': dataInterpMethod.BIQUINTIC_METHOD
    }

    def __cinit__(self, x=None, y=None, z=None, method='bilinear'):

        # Create C++ data
        cdef int i, N
        cdef valarray[double] x_array, y_array
        cdef Matrix[double] zmat

        if x is not None and y is not None and z is not None:
            x_array = numpyToValarray(x)
            y_array = numpyToValarray(y)
            zmat = numpyToMatrix(z)
            self.c_lut = new LUT2d[double](x_array, y_array, zmat, self.interpMethods[method])

        else:
            self.c_lut = new LUT2d[double]()

        self.__owner = True
        
    def __dealloc__(self):
        if self.__owner:
            del self.c_lut

    @staticmethod
    def bind(pyLUT2d lut):
        '''
        Creates a pyLUT2d object that acts as a reference to an existing
        pyLUT2d instance.
        '''
        new_lut = pyLUT2d()
        del new_lut.c_lut
        new_lut.c_lut = lut.c_lut
        new_lut.__owner = False
        return new_lut

    @staticmethod
    cdef cbind(LUT2d[double] c_lut):
        '''
        Creates a pyLUT2d object that creates a copy of a C++ LUT2d object.
        '''
        new_lut = pyLUT2d()
        del new_lut.c_lut
        new_lut.c_lut = new LUT2d[double](c_lut)
        new_lut.__owner = True
        return new_lut

    def eval(self, x, y):
        '''
        Evaluate LUT at given coordinate(s).

        Args:
            x (ndarray or float): X-coordinate(s) to evaluate at.
            y (ndarray or float): Y-coordiante(s) to evaluate at.

        Returns:
            ndarray or float : Value(s) of LUT at coordinates
        '''
        # Initialize numpy arrays
        cdef np.ndarray[double, ndim=1] x_np = np.array(x).squeeze()
        cdef np.ndarray[double, ndim=1] y_np = np.array(y).squeeze()
        cdef np.ndarray[double, ndim=1] values = np.empty_like(x_np, dtype=np.float64) 

        # Call interpolator for all points
        cdef int N = x_np.shape[0]
        cdef int i
        for i in range(N):
            values[i] = self.c_lut.eval(x_np[i], y_np[i])

        return values

    def __call__(self, x, y):
        '''
        Numpy-like interface to evaluate LUT.

        Args:
            x (ndarray or float): X-coordinate(s) to evaluate at.
            y (ndarray or float): Y-coordiante(s) to evaluate at.

        Returns:
            ndarray or float : Value(s) of LUT at coordinates
        '''
        return self.eval(x, y)
    
# end of file 
