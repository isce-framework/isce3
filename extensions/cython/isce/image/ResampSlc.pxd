#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from libcpp.string cimport string

from Poly2d cimport Poly2d
from Product cimport Product
from ImageMode cimport ImageMode

cdef extern from "isce/image/ResampSlc.h" namespace "isce::image":

    # ResampSlc class
    cdef cppclass ResampSlc:

        # Default constructor
        ResampSlc(const Product & product) except +

        # Polynomial getters
        Poly2d rgCarrier()
        Poly2d azCarrier()
        Poly2d doppler()
        # Polynomial setters
        void rgCarrier(Poly2d &)
        void azCarrier(Poly2d &)
        void doppler(Poly2d &)

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

        # Alternative generic resamp entry point
        void resamp(const string &, const string &, const string &,
                    const string &, int, bool, bool, int)

# end of file
