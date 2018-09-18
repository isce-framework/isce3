#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from ImageMode cimport ImageMode

cdef extern from "isce/product/ComplexImagery.h" namespace "isce::product":

    # Metadata class
    cdef cppclass ComplexImagery:

        # Constructors
        ComplexImagery() except +

        # Auxiliary mode
        ImageMode auxMode()
        void auxMode(const ImageMode &)

        # Primary mode
        ImageMode primaryMode()
        void primaryMode(const ImageMode &)

# end of file
