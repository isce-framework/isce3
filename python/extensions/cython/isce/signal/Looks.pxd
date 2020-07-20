#cython: language_level=3

from Raster cimport Raster

cdef extern from "isce3/signal/Looks.h" namespace "isce::signal":

    # Class definition
    cdef cppclass Looks[T]:

        # Constructors
        Looks(size_t nlooks_rg, size_t n_loos_az)
        void multilook(Raster & input_raster, Raster & output_raster, int p)
