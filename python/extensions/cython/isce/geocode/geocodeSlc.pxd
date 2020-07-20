# cython: language_level=3

from Product cimport Product
from RadarGridParameters cimport RadarGridParameters
from GeoGridParameters cimport GeoGridParameters

from Orbit cimport Orbit
from LUT2d cimport LUT2d

from Ellipsoid cimport Ellipsoid

from Raster  cimport Raster
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "isce3/geocode/geocodeSlc.h" namespace "isce::geocode":
    void geocodeSlc(Raster & outputRaster,
        Raster & inputRaster,
        Raster & demRaster,
        const RadarGridParameters & radarGrid,
        const GeoGridParameters & geoGrid,
        const Orbit& orbit,
        const LUT2d[double]& nativeDoppler,
        const LUT2d[double]& imageGridDoppler,
        const Ellipsoid & ellipsoid,
        const double & thresholdGeo2rdr,
        const int & numiterGeo2rdr,
        const size_t & linesPerBlock,
        const double & demBlockMargin,
        const bool flatten)

