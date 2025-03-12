//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

#pragma once

#include <ogr_geometry.h>
#include <cmath>

namespace isce3 { namespace geometry {
    /** Same as GDAL's OGRLinearRing structure. See: https://gdal.org/doxygen/classOGRLinearRing.html */
    typedef OGRLinearRing Perimeter;

    /** Extend GDAL's OGREnvelope to account for dateline crossing case in geographic coordinates.
        See: https://gdal.org/doxygen/ogr__core_8h_source.html */
    class BoundingBox : public OGREnvelope {
        public:
        void Merge2(const BoundingBox& other, int epsg=-1) {
            const double one_cycle = 360.0;

            double minx_global = (MinX < other.MinX) ? MinX : other.MinX;
            double maxx_global = (MaxX > other.MaxX) ? MaxX : other.MaxX;

            if ((epsg != 4326) || (maxx_global - minx_global) <= 180.0){
                // just use the method in the base class
                Merge(other);
                return;
            }

            // compute the "wrapped" x coordinates
            double minx_wrapped_this = std::fmod(MinX + one_cycle, one_cycle);
            double maxx_wrapped_this = std::fmod(MaxX + one_cycle, one_cycle);

            double minx_wrapped_other = std::fmod(other.MinX + one_cycle, one_cycle);
            double maxx_wrapped_other = std::fmod(other.MaxX + one_cycle, one_cycle);

            MinX = (minx_wrapped_this < minx_wrapped_other) ? minx_wrapped_this : minx_wrapped_other;
            MaxX = (maxx_wrapped_this > maxx_wrapped_other) ? maxx_wrapped_this : maxx_wrapped_other;
            MinY = (MinY < other.MinY) ? MinY : other.MinY;
            MaxY = (MaxY > other.MaxY) ? MaxY : other.MaxY;

            return;
            }
        };
    }
    //typedef OGREnvelope BoundingBox;

    /** Same as GDAL's OGRTriangle structure. See: https://gdal.org/doxygen/classOGRTriangle.html */
    typedef OGRTriangle Triangle;
}

//end of file
