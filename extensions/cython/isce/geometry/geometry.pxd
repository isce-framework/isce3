#cython: language_level=3
#
# Author: Bryan Riel, Piyush Agram, Tamas Gal
# Copyright 2017-2019
#

from DEMInterpolator cimport DEMInterpolator
from Orbit cimport Orbit, orbitInterpMethod
from Ellipsoid cimport Ellipsoid
from Cartesian cimport cartesian_t
from LUT2d cimport LUT2d
from RadarGridParameters cimport RadarGridParameters

cdef extern from "isce/geometry/geometry.h" namespace "isce::geometry":

    # Map coordinates to radar geometry coordinates transformer
    int geo2rdr(const cartesian_t &,
                const Ellipsoid &,
                const Orbit &,
                const LUT2d[double] &,
                double &, double &,
                double, double, int, double)

    # Radar geometry coordinates to map coordinates transformer
    int rdr2geo(double, double, double,
                const Orbit &, const Ellipsoid &, const DEMInterpolator &,
                cartesian_t &,
                double, int, double, int, int, orbitInterpMethod)

    # Utility function to compute geographic bounds for a radar grid
    void computeDEMBounds(const Orbit & orbit,
                          const Ellipsoid & ellipsoid,
                          const LUT2d[double] & doppler,
                          int lookSide,
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
