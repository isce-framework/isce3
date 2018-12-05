#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string

from LUT1d cimport LUT1d
from Poly2d cimport Poly2d
from Product cimport Product
from ImageMode cimport ImageMode
from Raster cimport Raster

cdef extern from "isce/image/ResampSlc.h" namespace "isce::image":

    # ResampSlc class
    cdef cppclass ResampSlc:

        # Default constructor
        ResampSlc() except +
        # Constructor with a Product
        ResampSlc(const Product & product) except +
        # Constructor with Doppler and ImageMode
        ResampSlc(const LUT1d[double] & doppler, const ImageMode & mode) except +

        # Polynomial getters
        Poly2d rgCarrier()
        Poly2d azCarrier()
        LUT1d[double] doppler()
        # Polynomial setters
        void rgCarrier(Poly2d &)
        void azCarrier(Poly2d &)
        void doppler(LUT1d[double] &)

        # Set reference product
        void referenceProduct(const Product & refProduct)

        # Get image modes
        ImageMode imageMode()
        ImageMode refImageMode()
        # Set metadata
        void imageMode(const ImageMode &)
        void refImageMode(const ImageMode &)

        # Get/set number of lines per processing tile
        size_t linesPerTile()
        void linesPerTile(size_t)

        # Main product-based resamp entry point
        void resamp(const string &, const string &, const string &,
                    const string &, bool, bool, int)

        # Generic resamp entry point: use filenames to create rasters
        void resamp(const string &, const string &, const string &,
                    const string &, int, bool, bool, int)

        # Generic resamp entry point from externally created rasters
        void resamp(Raster &, Raster &, Raster &, Raster &,
                    int, bool, bool, int)

# end of file
