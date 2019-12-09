#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018

from libcpp.string cimport string

# Wrapper around isce::geometry::load_archive in <isce/geometry/Serialization.h
cdef extern from "isce/geometry/Serialization.h" namespace "isce::geometry":
    void load_archive[T](string metadata, char * objTag, T * obj)

# end of file
