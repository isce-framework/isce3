#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Poly1d cimport Poly1d
import numpy as np
cimport numpy as np

cdef class pyPoly1d:
    '''
    Python Wrapper for isce::core::Poly1d

    Note:
        Order of polynomial must be set before setting coefficients

    Args:
        order (Optional[int]): Order of the polynomial
        mean (Optional[float]): Mean value for normalization
        norm (Optional[float]): Scale value for normalization
    '''

    #Pointer to underlying C++ object
    cdef Poly1d *c_poly1d

    #Variable that tracks ownership of underlying C++ object
    cdef bool __owner

    #Constructor
    def __cinit__(self, order=-1, mean=0., norm=1.):
        self.c_poly1d = new Poly1d(order,mean,norm)
        self.__owner = True

    #Destructor
    def __dealloc__(self):
        if self.__owner:
            del self.c_poly1d

    #Method to bind python wrapper to existing C++ pointer
    @staticmethod
    def bind(pyPoly1d poly):
        new_poly = pyPoly1d()
        del new_poly.c_poly1d
        new_poly.c_poly1d = poly.c_poly1d
        new_poly.__owner = False
        return new_poly

    #Interface to polynomial order
    @property
    def order(self):
        '''
        Return order of the polynomial.

        Returns:
            int: Order of polynomial
        '''
        return self.c_poly1d.order

    @order.setter
    def order(self, int order):
        '''
        Set the order of the polynomial.
        
        Args:
            order (int): Order of polynomial
        '''
        if (order < 0):
            raise ValueError('Order of a polynomial cannot be negative: {0}'.format(order))

        self.c_poly1d.order = order
        self.c_poly1d.coeffs.resize(order+1)

    #Interface to polynomial mean
    @property
    def mean(self):
        '''
        Return mean value used for normalizing inputs.

        Returns:
            float : Mean value
        '''
        return self.c_poly1d.mean

    @mean.setter
    def mean(self, double mean):
        '''
        Set mean value used for normalizing inputs.

        Args:
            mean (float): Mean value
        '''
        self.c_poly1d.mean = mean

    #Interface to polynomial norm
    @property
    def norm(self):
        '''
        Return scale value used for normalizing inputs.

        Returns:
            float : norm value
        '''
        return self.c_poly1d.norm

    @norm.setter
    def norm(self, double norm):
        '''
        Set the scale value used for normalizing inputs.

        Args:
            norm (float): norm value
        '''
        self.c_poly1d.norm = norm

    #Memory view of C++ vector member
    @property
    def coeffs(self):
        '''
        Return a numpy view of underlying C++ array for coefficients.

        Note:
            This method returns reference to actual C++ data. Modifying contents at python level will impact the C++ level data structure. Alternately, destroying the source poly1D python object will also impact the numpy array.

        Returns:
            numpy.array : Array of coefficients.
        '''
        cdef ssize_t N = self.c_poly1d.coeffs.size()
        res = np.asarray(<double[:N]>self.c_poly1d.coeffs.data())
        return res

    def copy(self, poly):
        '''
        Utility method for copying metadata from another python object.

        Args:
            poly (object): Any python object that has attributes like pyPoly1D

        Returns:
            None
        '''
        try:
            self.order = poly.order
            self.mean = poly.mean
            self.norm = poly.norm
            self.coeffs = poly.coeffs
        except:
            raise ValueError("bject passed in to copy is incompatible with object of type pyPoly1d.")

    #Interface to C++ Set function. 
    def setCoeff(self, int index, double val):
        '''
        Set coefficient at given index.

        Args:
            index (int): Index into the array of coefficients
            val (float): Value of coefficient

        Returns:
            None
        '''
        self.c_poly1d.setCoeff(index, val)

    #Interface to C++ Get function.
    def getCoeff(self, int index):
        '''
        Get coefficient at given index.

        Args:
            index(int): Index into array of coefficients

        Returns:
            float : Value of coefficient
        '''
        return self.c_poly1d.getCoeff(index)

    #Interface to C++ eval function.
    def eval(self, x):
        '''
        Evaluate function at a given value.

        Args:
            x (float or np.array): Value to evaluate polynomial at.

        Returns:
            float or np.array: Value of polynomial at x.
        '''
        if np.isscalar(x):
            return self.c_poly1d.eval(x)

        x = np.atleast_1d(x)
        cdef unsigned long nPts = x.shape[0]
        res = np.empty(nPts, dtype=np.double)
        cdef double[:] resview = res
        cdef unsigned long ii
        for ii in range(nPts):
            resview[ii] = self.c_poly1d.eval(x[ii])
        return res

    #Interface to C++ print function.
    def printPoly(self):
        '''
        Print poly structure at C++ level for debugging.

        Returns:
            None
        '''
        self.c_poly1d.printPoly()

    #Numpy like interface
    def __call__(self, x):
        '''
        Numpy-like interface to eval function.
        '''
        return self.eval(x)

    #Evaluate using numpy
    def evalWithNumpy(self, x):
        '''
        Evaluate using numpy's polyval method
        '''
        p = self.coeffs()
        return np.polyval(p[::-1], (x-self.mean)/self.order)
