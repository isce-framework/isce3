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
    cdef pyMetadata pyMeta
    cdef pyMetadata pyRefMeta

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

    # Metadata
    @property
    def metadata(self):
        return self.pyMeta
    @metadata.setter 
    def metadata(self, pyMetadata meta):
        self.pyMeta = meta
        self.c_resamp.metadata(meta.c_metadata)

    # Reference metadata
    @property
    def refMetadata(self):
        return self.pyRefMeta
    @refMetadata.setter 
    def refMetadata(self, pyMetadata meta):
        self.pyRefMeta = meta
        self.c_resamp.refMetadata(meta.c_metadata)

    # Get/set number of lines per processing tile
    @property
    def linesPerTile(self):
        return self.c_resamp.linesPerTile()
    @linesPerTile.setter
    def linesPerTile(self, int lines):
        self.c_resamp.linesPerTile(lines)

    # Run resamp
    def resamp(self, infile, outfile, rgfile, azfile, bool flatten=True, bool isComplex=True,
               int rowBuffer=40):
        """
        Run resamp.
        """
        cdef string inputfile = pyStringToBytes(infile)
        cdef string outputfile = pyStringToBytes(outfile)
        cdef string rgofffile = pyStringToBytes(rgfile)
        cdef string azofffile = pyStringToBytes(azfile)
        self.c_resamp.resamp(inputfile, outputfile, rgofffile, azofffile,
                             flatten, isComplex, rowBuffer)
    
# end of file
