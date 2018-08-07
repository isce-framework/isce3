#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017
#

from libcpp cimport bool
from libcpp.string cimport string
from ResampSlc cimport ResampSlc

cdef class pyResampSlc:
    # C++ class pointer
    cdef ResampSlc c_resamp
    
    # Cython classes
    cdef pyPoly2d pyDop
    cdef pyImageMode pyImageMode
    cdef pyImageMode pyRefImageMode

    def __cinit__(self):
        self.c_resamp = ResampSlc()

    # Doppler
    @property
    def doppler(self):
        return self.pyDop
    @doppler.setter
    def doppler(self, pyPoly2d dop):
        self.pyDop = dop
        self.c_resamp.doppler(deref(dop.c_poly2d))

    # ImageMode
    @property
    def imageMode(self):
        return self.pyImageMode
    @imageMode.setter 
    def imageMode(self, pyImageMode mode):
        self.pyImageMode = mode
        self.c_resamp.imageMode(deref(mode.c_imagemode))

    # Reference metadata
    @property
    def refImageMode(self):
        return self.pyRefImageMode
    @refImageMode.setter 
    def refImageMode(self, pyImageMode mode):
        self.pyRefImageMode = mode
        self.c_resamp.refImageMode(deref(mode.c_imagemode))

    # Get/set number of lines per processing tile
    @property
    def linesPerTile(self):
        return self.c_resamp.linesPerTile()
    @linesPerTile.setter
    def linesPerTile(self, int lines):
        self.c_resamp.linesPerTile(lines)

    # Run resamp
    def resamp(self, infile, outfile, rgfile, azfile, int inputBand=1,
               bool flatten=True, bool isComplex=True, int rowBuffer=40):
        """
        Run resamp.
        """
        cdef string inputFile = pyStringToBytes(infile)
        cdef string outputFile = pyStringToBytes(outfile)
        cdef string rgoffFile = pyStringToBytes(rgfile)
        cdef string azoffFile = pyStringToBytes(azfile)
        self.c_resamp.resamp(inputFile, outputFile, rgoffFile, azoffFile,
                             inputBand, flatten, isComplex, rowBuffer)
    
# end of file
