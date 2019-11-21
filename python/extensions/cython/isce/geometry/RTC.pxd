# cython: language_level=3

from Product cimport Product
from Raster  cimport Raster

cdef extern from "isce/core/Constants.h" namespace "isce::core":
    cdef enum rtcInputRadiometry:
        BETA_NAUGHT = 0
        SIGMA_NAUGHT = 1

cdef extern from "isce/core/Constants.h" namespace "isce::core":
    cdef enum rtcOutputMode:
        GAMMA_NAUGHT_AREA = 0
        GAMMA_NAUGHT_DIVISOR = 1

cdef extern from "isce/geometry/RTC.h" namespace "isce::geometry":
    void facetRTC(Product& product, Raster& dem, Raster& out_raster,
                  char frequency,
                  rtcInputRadiometry inputRadiometry,
                  rtcOutputMode outputMode)
