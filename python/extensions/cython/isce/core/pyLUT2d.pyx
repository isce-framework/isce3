#cython: language_level=3
#
# Author: Bryan Riel, Joshua Cohen
# Copyright 2017-2018
#

from libcpp cimport bool
import numpy as np
cimport numpy as np
from LUT2d cimport LUT2d, loadCalGrid, saveCalGrid
from IH5 cimport hid_t, IGroup

cdef class pyLUT2d:
    '''
    Python wrapper for isce::core::LUT2d

    KwArgs:
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
        'biquintic': dataInterpMethod.BIQUINTIC_METHOD
    }

    def __cinit__(self, x=None, y=None, z=None, method='bilinear'):

        # Create C++ data
        cdef int i, N
        cdef valarray[double] x_array, y_array
        cdef Matrix[double] zmat
        cdef np.ndarray[double, ndim=2] znp

        if x is not None and y is not None and z is not None:
            x_array = numpyToValarray(x)
            y_array = numpyToValarray(y)
            # FIXME see note in pyInterpolator.numpyToMatrix
            # FIXME zmat = numpyToMatrix(z)
            znp = np.asarray(z, 'f8')
            zmat = Matrix[double](&znp[0,0], len(y), len(x))
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

    def eval(self, y, x):
        '''
        Evaluate LUT at given coordinate(s).

        Args:
            y (ndarray or float): Y-coordiante(s) to evaluate at.
            x (ndarray or float): X-coordinate(s) to evaluate at.

        Returns:
            ndarray or float : Value(s) of LUT at coordinates
        '''
        # Initialize numpy arrays
        cdef np.ndarray[double, ndim=1] x_np = np.array(x, 'f8').flatten()
        cdef np.ndarray[double, ndim=1] y_np = np.array(y, 'f8').flatten()
        cdef np.ndarray[double, ndim=1] values = np.empty_like(x_np, dtype=np.float64) 
        # TODO handle scalar x (or y) and vector y (or x).
        assert len(x_np) == len(y_np)

        # Call interpolator for all points
        cdef int N = len(x_np)
        cdef int i
        for i in range(N):
            values[i] = self.c_lut.eval(y_np[i], x_np[i])

        if N > 1:
            return values
        return values[0]

    def __call__(self, y, x):
        '''
        Numpy-like interface to evaluate LUT.

        Args:
            y (ndarray or float): Y-coordiante(s) to evaluate at.
            x (ndarray or float): X-coordinate(s) to evaluate at.

        Returns:
            ndarray or float : Value(s) of LUT at coordinates
        '''
        return self.eval(y, x)
    
    @property
    def xStart(self):
        return self.c_lut.xStart()

    @staticmethod
    def loadCalGrid(h5Group, dsetName):
        cdef hid_t groupid = h5Group.id.id
        cdef IGroup c_igroup
        c_igroup = IGroup(groupid)
        lutObj = pyLUT2d()
        loadCalGrid(c_igroup, <string> dsetName, deref(lutObj.c_lut))

    def saveCalGrid(self, h5Group, dsetName, pyDateTime refEpoch, units=""):
        cdef hid_t groupid = h5Group.id.id
        cdef IGroup c_igroup
        c_igroup = IGroup(groupid)
        saveCalGrid(c_igroup, <string> dsetName.encode("UTF-8"), deref(self.c_lut),
                    deref(refEpoch.c_datetime), <string> units.encode("UTF-8"))
