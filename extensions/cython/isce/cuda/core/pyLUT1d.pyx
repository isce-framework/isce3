#cython: language_level=3
#
# Author: Bryan Riel, Joshua Cohen
# Copyright 2017-2018
#

from libcpp cimport bool
import numpy as np
cimport numpy as np
from LUT1d cimport LUT1d

cdef class pyLUT1d:
    '''
    Python wrapper for isce::core::LUT1d

    Args:
        x (ndarray or list): Coordinates for LUT
        y (ndarray or list): Values for LUT
        extrapolate (Optional[bool]): Flag for allowing extrapolation beyond bounds
    '''
    cdef LUT1d[double] * c_lut
    cdef bool __owner

    def __cinit__(self, x=None, y=None, bool extrapolate=False):

        cdef int i, N
        cdef np.ndarray[double, ndim=1] x_np, y_np
        cdef valarray[double] x_array, y_array

        if x is not None and y is not None:

            # Make sure coordinates and values are numpy arrays
            x_np = np.array(x).squeeze()
            y_np = np.array(y).squeeze()

            # Copy to valarrays
            N = x_np.shape[0]
            x_array = valarray[double](N)
            y_array = valarray[double](N)
            for i in range(N):
                x_array[i] = x[i]
                y_array[i] = y[i]

            # Instantiate LUT1d with coordinates and values
            self.c_lut = new LUT1d[double](x_array, y_array, extrapolate)
 
        else:
            # Instantiate default LUT1d
            self.c_lut = new LUT1d[double]()

        return

    def __dealloc__(self):
        if self.__owner:
            del self.c_lut

    @staticmethod
    def bind(pyLUT1d lut):
        '''
        Creates a pyLUT1d object that acts as a reference to an existing
        pyLUT1d instance.
        '''
        new_lut = pyLUT1d()
        del new_lut.c_lut
        new_lut.c_lut = lut.c_lut
        new_lut.__owner = False
        return new_lut

    @staticmethod
    cdef cbind(LUT1d[double] c_lut):
        '''
        Creates a pyLUT1d object that creates a copy of a C++ LUT1d object.
        '''
        new_lut = pyLUT1d()
        del new_lut.c_lut
        new_lut.c_lut = new LUT1d[double](c_lut)
        new_lut.__owner = True
        return new_lut

    def eval(self, x):
        '''
        Evaluate LUT at given coordinate(s).

        Args:
            x (ndarray or float): Coordinate(s) to evaluate at.

        Returns:
            float : Value of LUT at x
        '''
        # Initialize numpy arrays
        cdef np.ndarray[double, ndim=1] x_np = np.array(x).squeeze()
        cdef np.ndarray[double, ndim=1] values = np.empty_like(x_np, dtype=np.float64) 

        # Call interpolator for all points
        cdef int N = x_np.shape[0]
        cdef int i
        for i in range(N):
            values[i] = self.c_lut.eval(x_np[i])

        return values

    def __call__(self, x):
        '''
        Numpy-like interface to evaluate LUT.

        Args:
            x (ndarray or float): Coordinate(s) to evaluate at.

        Returns:
            float : Value of LUT at x
        '''
        return self.eval(x)
    
# end of file 
