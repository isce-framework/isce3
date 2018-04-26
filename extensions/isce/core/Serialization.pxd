#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018

from libcpp.string cimport string

# Wrapper around isce::core::load_archive in <isce/core/Serialization.h
cdef extern from "isce/core/Serialization.h" namespace "isce::core":
    void load_archive[T](string metadata, char * objTag, T * obj)
    void load_archive_reference[T](string metadata, char * objTag, T & obj)

# end of file
