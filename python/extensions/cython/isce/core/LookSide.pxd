#cython: language_level=3
from libcpp.string cimport string

cdef extern from "isce3/core/LookSide.h" namespace "isce3::core":

    cdef enum LookSide:
        Left "isce3::core::LookSide::Left"
        Right "isce3::core::LookSide::Right"

    string to_string(LookSide d)
    LookSide parseLookSide(string s)
