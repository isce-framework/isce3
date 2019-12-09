//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

#pragma once 

#include <ogr_geometry.h>

namespace isce { namespace geometry {
    /** Same as GDAL's OGRLinearRing structure. See: https://gdal.org/doxygen/classOGRLinearRing.html */
    typedef OGRLinearRing Perimeter;

    /** Same as GDAL's OGREnvelope structure. See: https://gdal.org/doxygen/ogr__core_8h_source.html */
    typedef OGREnvelope BoundingBox;

    /** Same as GDAL's OGRTriangle structure. See: https://gdal.org/doxygen/classOGRTriangle.html */
    typedef OGRTriangle Triangle;
}}

//end of file
