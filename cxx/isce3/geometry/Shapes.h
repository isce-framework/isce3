//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

#pragma once

#include <ogr_geometry.h>
#include <cmath>
#include <algorithm>

namespace isce3 { namespace geometry {
    /** Same as GDAL's OGRLinearRing structure. See: https://gdal.org/doxygen/classOGRLinearRing.html */
    typedef OGRLinearRing Perimeter;

    /** Extend GDAL's OGREnvelope to account for dateline crossing case in geographic coordinates.
        See: https://gdal.org/doxygen/ogr__core_8h_source.html */
    class BoundingBox : public OGREnvelope {
        public:
        /*Same as OGREnvelope::Merge but account for dateline crossing case
        in geographic coordinates.*/
        void Merge2(const BoundingBox& other, int epsg=-1) {
            const double one_cycle = 360.0;

            double minx_global = (MinX < other.MinX) ? MinX : other.MinX;
            double maxx_global = (MaxX > other.MaxX) ? MaxX : other.MaxX;

            // Check if wrapping is necessary. If not, use the method in the base class
            if ((epsg != 4326) || (maxx_global - minx_global) <= 180.0){
                // just use the method in the base class
                Merge(other);
                return;
            }

            // compute the "wrapped" x coordinates
            auto wrap = [one_cycle](double angle) { return std::fmod(angle + one_cycle, one_cycle); };

            MinX = std::min(wrap(MinX), wrap(other.MinX));
            MaxX = std::max(wrap(MaxX), wrap(other.MaxX));
            MinY = std::min(MinY, other.MinY);
            MaxY = std::max(MaxY, other.MaxY);

            return;
            }
        };
    }
    //typedef OGREnvelope BoundingBox;

    /** Same as GDAL's OGRTriangle structure. See: https://gdal.org/doxygen/classOGRTriangle.html */
    typedef OGRTriangle Triangle;
}

//end of file
