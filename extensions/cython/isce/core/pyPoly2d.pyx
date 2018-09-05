#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp cimport bool
from Poly2d cimport Poly2d

cdef class pyPoly2d:
    '''
    Python wrapper for isce::core::Poly2D

    Note:
        Always set the orders of the polynomials before assigning coefficients to it.

    Args:
        azimuthOrder (Optional[int]): Order of polynomial in azimuth (y)
        rangeOrder (Optional[int]): Order of polynomial in range (x)
        azimuthMean (Optional[float]): Mean for normalizing value of input in azimuth
        rangeMean (Optional[float]): Mean for normalizing value of input in range
        azimuthNorm (Optional[float]): Scale for normalizing value of input in azimuth
        rangeNorm (Optional[float]): Scale for normalizing value of input in range
    '''

    cdef Poly2d *c_poly2d
    cdef bool __owner

    def __cinit__(self, int azimuthOrder=-1, int rangeOrder=-1, double azimuthMean=0., 
                  double rangeMean=0., double azimuthNorm=1., double rangeNorm=1.):
        self.c_poly2d = new Poly2d(azimuthOrder,rangeOrder,azimuthMean,rangeMean,azimuthNorm,
                                    rangeNorm)
        self.__owner = True


    def __dealloc__(self):
        if self.__owner:
            del self.c_poly2d


    @staticmethod
    def bind(pyPoly2d poly):
        '''
        Creates a pyPoly2d object that acts as a reference to an existing
        pyPoly2d instance.
        '''
        new_poly = pyPoly2d()
        del new_poly.c_poly2d
        new_poly.c_poly2d = poly.c_poly2d
        new_poly.__owner = False
        return new_poly

    @staticmethod
    cdef cbind(Poly2d c_poly2d):
        '''
        Creates a pyPoly2d object that creates a copy of a C++ Poly2d object.
        '''
        new_poly = pyPoly2d()
        del new_poly.c_poly2d
        new_poly.c_poly2d = new Poly2d(c_poly2d)
        new_poly.__owner = True
        return new_poly

    @property
    def azimuthOrder(self):
        '''
        Returns azimuth order.
    
        Returns:
            int : Azimuth order of polynomial
        '''
        return self.c_poly2d.azimuthOrder

    @azimuthOrder.setter
    def azimuthOrder(self, int order):
        '''
        Sets azimuth order.

        Args:
            order (int): azimuth order of polynomial.
        '''

        if (order < 0):
            raise ValueError('Azimuth order of polynomial cannot be negative: {0}'.format(order))
        else:
            c = self.coeffs
            for ii in range((order-self.azimuthOrder)*(self.rangeOrder+1)):
                c.append(0.)

            nc = []
            for ii in range((order+1)*(self.rangeOrder+1)):
                nc.append(c[ii])

            self.c_poly2d.azimuthOrder = order
            self.c_poly2d.coeffs.resize((self.azimuthOrder+1)*(self.rangeOrder+1))
            self.coeffs = nc

    @property
    def rangeOrder(self):
        '''
        Return the range order of polynomial.

        Returns:
            int: Range order of polynomial.
        '''
        return self.c_poly2d.rangeOrder

    @rangeOrder.setter
    def rangeOrder(self, int order):
        '''
        Sets range order.

        Args:
            order (int): range order of polynomial.
        '''
        if (order < 0):
            raise ValueError('Range order of polynomial cannot be negative: {0}'.format(order))
        else:
            c = self.coeffs
            nc = []
            # Cleanest is to first form 2D array of coeffs from 1D
            for ii in range(self.azimuthOrder+1):
                ncs = []
                for jj in range(self.rangeOrder+1):
                    ncs.append(c[ii*(self.rangeOrder+1)+jj])
                nc.append(ncs)
            # nc is now the 2D reshape of coeffs
            # Go row-by-row...
            for ii in range(self.azimuthOrder+1):
                # Add 0s to each row (if a > self.rangeOrder)
                for jj in range(order-self.rangeOrder):
                    nc[ii].append(0.)
            self.c_poly2d.rangeOrder = order
            self.c_poly2d.coeffs.resize((self.azimuthOrder+1)*(self.rangeOrder+1))
            c = []
            for ii in range(self.azimuthOrder+1):
                for jj in range(self.rangeOrder+1):
                    c.append(nc[ii][jj])
            self.coeffs = c

    @property
    def azimuthMean(self):
        '''
        Return azimuth mean.

        Returns:
            float: azimuth mean of the polynomial.
        '''
        return self.c_poly2d.azimuthMean

    @azimuthMean.setter
    def azimuthMean(self, double mean):
        '''
        Set azimuth mean.

        Args:
            mean (float): azimuth mean.
        '''
        self.c_poly2d.azimuthMean = mean

    @property
    def rangeMean(self):
        '''
        Return range mean.

        Returns:
            float : range mean
        '''
        return self.c_poly2d.rangeMean

    @rangeMean.setter
    def rangeMean(self, double mean):
        '''
        Set range mean

        Args:
            mean (float): range mean
        '''
        self.c_poly2d.rangeMean = mean


    @property
    def azimuthNorm(self):
        '''
        Return azimuth norm.

        Returns:
            float: azimuth norm
        '''
        return self.c_poly2d.azimuthNorm

    @azimuthNorm.setter
    def azimuthNorm(self, double norm):
        '''
        Set azimuth norm

        Args:
            norm (float): azimuth norm
        '''
        self.c_poly2d.azimuthNorm = norm

    @property
    def rangeNorm(self):
        '''
        Return range norm

        Returns:
            float: range norm
        '''
        return self.c_poly2d.rangeNorm

    @rangeNorm.setter
    def rangeNorm(self, double norm):
        '''
        Set range norm

        Args:
            norm (float): range norm
        '''
        self.c_poly2d.rangeNorm = norm
    
    @property
    def coeffs(self):
        '''
        Return polynomial coefficients.

        Note:
            This method returns an actual reference to the array of coefficients. Modifying contents will affect data at C++ level and deleting the C++ datastructure will impact the returned array.
            
        Returns:
            numpy.array : List of coefficients
        '''
        cdef int Ny = self.azimuthOrder + 1 
        cdef int Nx = self.rangeOrder + 1 
        res = np.asarray(<double[:Ny*Nx]>(self.c_poly2d.coeffs.data()))
        return res

    @coeffs.setter
    def coeffs(self, arr):
        '''
        Set polynomial coefficients using a list / array.

        Args:
            arr (list): List of coefficients
        '''
        if ((self.azimuthOrder+1)*(self.rangeOrder+1) != len(arr)):
            ValueError("Invalid input size (expected 1D list of length "+str(self.azimuthOrder+1)+
                  "*"+str(self.rangeOrder+1)+")")
        for ii in range((self.azimuthOrder+1)*(self.rangeOrder+1)):
            self.c_poly2d.coeffs[ii] = arr[ii]
    
    def dPrint(self):
        '''
        Debug print of underlying C++ structure.

        Returns: 
            None
        '''
        self.printPoly()    

    def setCoeff(self, int row, int col, double val):
        '''
        Set polynomial coefficient using row and col indices.

        Args:
            row (int): Index in azimuth direction
            col (int): Index in range direction
            val (float): Value of coefficient

        Returns:
            None
        '''
        self.c_poly2d.setCoeff(row, col, val)

    def getCoeff(self, int row, int col):
        '''
        Get polynomial coefficient using row and col indices.

        Args:
            row (int): Index in azimuth direction
            col (int): Index in range direction

        Returns:
            float: Value of coefficient
        '''
        return self.c_poly2d.getCoeff(row,col)


    def eval(self, double azi, double rng):
        '''
        Evaluate polynomial at given point.

        Args:
            azi (float): Azimuth coordinate
            rng (float): Range coordinate

        Returns:
            float : Value of polynomial at (azi, rng)
        '''
        return self.c_poly2d.eval(azi,rng)

    def __call__(self, azi, rng):
        '''
        Numpy-like interface to evaluate polynomial.

        Args:
            azi (int): Azimuth coordinate
            rng (int): Range coordinate

        Returns:
            float: Value of polynomial at (azi, rng)
        '''
        return self.eval(azi, rng)
    
    def printPoly(self):
        '''
        Debug print function of underlying C++ structure.
        '''
        self.c_poly2d.printPoly()

# end of file 
