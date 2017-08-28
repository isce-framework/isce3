#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from Poly2d cimport Poly2d

cdef class pyPoly2d:
    cdef Poly2d c_poly2d

    def __cinit__(self, int azimuthOrder=-1, int rangeOrder=-1, double azimuthMean=0., 
                  double rangeMean=0., double azimuthNorm=1., double rangeNorm=1.):
        self.c_poly2d.azimuthOrder = azimuthOrder
        self.c_poly2d.rangeOrder = rangeOrder
        self.c_poly2d.azimuthMean = azimuthMean
        self.c_poly2d.rangeMean = rangeMean
        self.c_poly2d.azimuthNorm = azimuthNorm
        self.c_poly2d.rangeNorm = rangeNorm
        self.c_poly2d.coeffs.resize((azimuthOrder+1)*(rangeOrder+1))
    
    @property
    def azimuthOrder(self):
        return self.c_poly2d.azimuthOrder
    @azimuthOrder.setter
    def azimuthOrder(self, int a):
        if (a < 0):
            return
        else:
            c = self.coeffs
            for i in range((a-self.azimuthOrder)*(self.rangeOrder+1)):
                c.append(0.)
            nc = []
            for i in range((a+1)*(self.rangeOrder+1)):
                nc.append(c[i])
            self.c_poly2d.azimuthOrder = a
            self.c_poly2d.coeffs.resize((self.azimuthOrder+1)*(self.rangeOrder+1))
            self.coeffs = nc
    @property
    def rangeOrder(self):
        return self.c_poly2d.rangeOrder
    @rangeOrder.setter
    def rangeOrder(self, int a):
        if (a < 0):
            return
        else:
            c = self.coeffs
            nc = []
            # Cleanest is to first form 2D array of coeffs from 1D
            for i in range(self.azimuthOrder+1):
                ncs = []
                for j in range(self.rangeOrder+1):
                    ncs.append(c[i*(self.rangeOrder+1)+j])
                nc.append(ncs)
            # nc is now the 2D reshape of coeffs
            # Go row-by-row...
            for i in range(self.azimuthOrder+1):
                # Add 0s to each row (if a > self.rangeOrder)
                for j in range(a-self.rangeOrder):
                    nc[i].append(0.)
            self.c_poly2d.rangeOrder = a
            self.c_poly2d.coeffs.resize((self.azimuthOrder+1)*(self.rangeOrder+1))
            c = []
            for i in range(self.azimuthOrder+1):
                for j in range(self.rangeOrder+1):
                    c.append(nc[i][j])
            self.coeffs = c
    @property
    def azimuthMean(self):
        return self.c_poly2d.azimuthMean
    @azimuthMean.setter
    def azimuthMean(self, double a):
        self.c_poly2d.azimuthMean = a
    @property
    def rangeMean(self):
        return self.c_poly2d.rangeMean
    @rangeMean.setter
    def rangeMean(self, double a):
        self.c_poly2d.rangeMean = a
    @property
    def azimuthNorm(self):
        return self.c_poly2d.azimuthNorm
    @azimuthNorm.setter
    def azimuthNorm(self, double a):
        self.c_poly2d.azimuthNorm = a
    @property
    def rangeNorm(self):
        return self.c_poly2d.rangeNorm
    @rangeNorm.setter
    def rangeNorm(self, double a):
        self.c_poly2d.rangeNorm = a
    @property
    def coeffs(self):
        a = []
        for i in range((self.azimuthOrder+1)*(self.rangeOrder+1)):
            a.append(self.c_poly2d.coeffs[i])
        return a
    @coeffs.setter
    def coeffs(self, a):
        if ((self.azimuthOrder+1)*(self.rangeOrder+1) != len(a)):
            print("Error: Invalid input size (expected 1D list of length "+str(self.azimuthOrder+1)+
                  "*"+str(self.rangeOrder+1)+")")
            return
        for i in range((self.azimuthOrder+1)*(self.rangeOrder+1)):
            self.c_poly2d.coeffs[i] = a[i]
    
    def dPrint(self):
        self.printPoly()    

    def setCoeff(self, int a, int b, double c):
        self.c_poly2d.setCoeff(a,b,c)
    def getCoeff(self, int a, int b):
        return self.c_poly2d.getCoeff(a,b)
    def eval(self, double a, double b):
        return self.c_poly2d.eval(a,b)
    def printPoly(self):
        self.c_poly2d.printPoly()


