#cython: language_level=3
#
# Author: Bryan Riel, Piyush Agram, Tamas Gal
# Copyright 2017-2019
#

from Basis cimport Basis
from DEMInterpolator cimport DEMInterpolator
from Orbit cimport Orbit
from Ellipsoid cimport Ellipsoid
from Cartesian cimport cartesian_t
from LUT2d cimport LUT2d
from RadarGridParameters cimport RadarGridParameters
from libcpp.string cimport string

cdef extern from "isce/core/Pixel.h" namespace "isce::core":
    cdef cppclass Pixel:
        Pixel()
        Pixel(double r, double r_sin_squint, size_t bin)
    

cdef extern from "isce/geometry/geometry.h" namespace "isce::geometry":
    
    cdef enum Direction "isce::geometry::Direction":
        Left "isce::geometry::Direction::Left"
        Right "isce::geometry::Direction::Right"

    string printDirection(Direction d)
    Direction parseDirection(string s)
    
    # Map coordinates to radar geometry coordinates transformer
    int geo2rdr(const cartesian_t &,
                const Ellipsoid &,
                const Orbit &,
                const LUT2d[double] &,
                double &, double &,
                double, int, double, int, double)

    # Radar geometry coordinates to map coordinates transformer
    int rdr2geo(double, double, double,
                const Orbit &, const Ellipsoid &, const DEMInterpolator &,
                cartesian_t &,
                double, Direction, double, int, int)
    
    int rdr2geo(const Pixel & pixel,
                const Basis & TCNbasis,
                const cartesian_t & pos,
                const cartesian_t & vel,
                const Ellipsoid & ellipsoid,
                const DEMInterpolator & demInterp,
                cartesian_t & targetLLH,
                Direction side, double threshold, int maxIter, int extraIter)


    # Utility function to compute geographic bounds for a radar grid
    void computeDEMBounds(const Orbit & orbit,
                          const Ellipsoid & ellipsoid,
                          const LUT2d[double] & doppler,
                          Direction lookSide,
                          const RadarGridParameters & radarGrid,
                          size_t xoff,
                          size_t yoff,
                          size_t xsize,
                          size_t ysize,
                          double margin,
                          double & min_lon,
                          double & min_lat,
                          double & max_lon,
                          double & max_lat)

# end of file
