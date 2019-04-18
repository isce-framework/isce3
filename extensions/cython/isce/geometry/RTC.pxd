# cython: language_level=3

from Product cimport Product
from Raster  cimport Raster

cdef extern from "isce/geometry/RTC.h" namespace "isce::geometry":
    void facetRTC(Product& product, Raster& dem, Raster& out_raster)
