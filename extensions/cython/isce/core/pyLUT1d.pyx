#cython: language_level=3
#
# Author: Bryan Riel, Joshua Cohen
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string
import numpy as np
cimport numpy as np
from LUT1d cimport LUT1d, loadLUT1d
from Matrix cimport valarray
import h5py
from IH5 cimport hid_t, IGroup

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
        cdef valarray[double] x_array, y_array

        if x is not None and y is not None:

            # Copy to valarrays
            N = x.size
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

        self.__owner = True

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

    @property
    def coordinates(self):
        '''
        Get coordinates of LUT.
        '''
        # Get valarray for coordinates
        cdef valarray[double] c = self.c_lut.coords()

        # Copy to numpy array
        cdef np.ndarray[double, ndim=1] c_np = np.zeros(self.size, dtype=np.float64)
        cdef int i
        for i in range(self.size):
            c_np[i] = c[i]

        return c_np

    @coordinates.setter
    def coordinates(self, coords):
        raise NotImplementedError('Cannot set coordinates yet.')

    @property
    def values(self):
        '''
        Get values of LUT.
        '''
        # Get valarray for coordinates
        cdef valarray[double] v = self.c_lut.values()

        # Copy to numpy array
        cdef np.ndarray[double, ndim=1] v_np = np.zeros(self.size, dtype=np.float64)
        cdef int i
        for i in range(self.size):
            v_np[i] = v[i]

        return v_np

    @values.setter
    def values(self, vals):
        raise NotImplementedError('Cannot set values yet.')

    @property
    def size(self):
        '''
        Get size of LUT.
        '''
        cdef int N = self.c_lut.size()
        return N

    @size.setter
    def size(self, s):
        raise NotImplementedError('Cannot overwrite LUT size.')

    @staticmethod
    def loadFromH5(h5Group, name_coords='r0', name_values='skewdc_values'):
        '''
        Load LUT1d from an HDF5 group

        Args:
            h5Group (h5py group): HDF5 group with lut1d data

        Returns:
            pyLUT1d object
        '''

        cdef hid_t groupid = h5Group.id.id
        cdef IGroup c_igroup
        c_igroup = IGroup(groupid)
        lutObj = pyLUT1d()
        loadLUT1d(c_igroup, deref(lutObj.c_lut), <string> pyStringToBytes(name_coords),
                <string> pyStringToBytes(name_values))

        return lutObj

# end of file 
