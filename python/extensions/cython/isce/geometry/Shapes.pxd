#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

#Get some data types from GDAL
cdef extern from "ogr_geometry.h":

    #OGRLinearRing
    cdef cppclass OGRLinearRing:
        char* exportToJson()


#Redefinition in shapes.h
cdef extern from "isce3/geometry/Shapes.h" namespace "isce3::geometry":
    ctypedef OGRLinearRing Perimeter
