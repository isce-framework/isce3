#cython: language_level=3
from libcpp.string cimport string

cdef extern from "isce/core/LookSide.h" namespace "isce::core":

    cdef enum LookSide:
        Left "isce::core::LookSide::Left"
        Right "isce::core::LookSide::Right"

    string to_string(LookSide d)
    LookSide parseLookSide(string s)
