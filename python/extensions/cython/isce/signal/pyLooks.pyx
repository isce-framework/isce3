#cython: language_level=3

# from libcpp cimport bool
from Looks cimport Looks
from libcpp.complex cimport complex as complex_t
from cython.operator cimport dereference as deref
cimport cython

cdef class pyLooksBase:
    '''
    Python wrapper for isce3::signal::Looks

    Args:

    '''
    cdef bool __owner


cdef class pyLooksDouble(pyLooksBase):
    cdef Looks[double] * c_looks

    def __cinit__(self,
                  size_t nlooks_rg,
                  size_t nlooks_az):

        self.c_looks = new Looks[double](nlooks_rg, nlooks_az)
        self.__owner = True

    # Run multilook
    def multilook(self,
                  pyRaster input_raster,
                  pyRaster output_raster,
                  int exponent=1):
        self.c_looks.multilook(deref(input_raster.c_raster), 
                               deref(output_raster.c_raster),
                               exponent)
    
    def __dealloc__(self):
        if self.__owner:
            del self.c_looks


cdef class pyLooksFloat(pyLooksBase):
    cdef Looks[float] * c_looks

    def __cinit__(self,
                  size_t nlooks_rg,
                  size_t nlooks_az):

        self.c_looks = new Looks[float](nlooks_rg, nlooks_az)
        self.__owner = True
                
    # Run multilook
    def multilook(self,
                  pyRaster input_raster,
                  pyRaster output_raster,
                  int exponent=1):
        self.c_looks.multilook(deref(input_raster.c_raster), 
                               deref(output_raster.c_raster), exponent)
    
    def __dealloc__(self):
        if self.__owner:
            del self.c_looks

