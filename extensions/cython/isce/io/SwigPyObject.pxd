#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

cdef extern from "isce/io/Constants.h" namespace "isce::io":

    # SwigPyObject class used for exposing swig pointers in Cython
    ctypedef struct SwigPyObject:
        void * ptr

# end of file
