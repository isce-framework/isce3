#cython: language_level=3
from libcpp.string cimport string

cdef extern from "isce/geometry/geometry.h" namespace "isce::geometry":
    
    cdef enum Direction:
        Left "isce::geometry::Direction::Left"
        Right "isce::geometry::Direction::Right"

    string printDirection(Direction d)
    Direction parseDirection(string s)
