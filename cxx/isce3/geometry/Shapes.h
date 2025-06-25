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

    /** Extend GDAL's OGREnvelope to account for antimeridian crossing case in geographic coordinates.
        See: https://gdal.org/doxygen/ogr__core_8h_source.html */
    class BoundingBox : public OGREnvelope {
        public:
        //Overload OGREnvelope::Merge by adding parameter for EPSG in geographic coordinates.
        void Merge(const BoundingBox& other, int epsg) {
            double minx_global = std::min(MinX, other.MinX);
            double maxx_global = std::max(MaxX, other.MaxX);

            // Check if wrapping is necessary. If not, use the method in the base class
            if ((epsg != 4326) || (maxx_global - minx_global) <= 180.0){
                // just use the method in the base class
                OGREnvelope::Merge(other);
                return;
            }

            // Wrap the angles to the range [0, 360) if either of the
            // bounding boxes crosses the antimeridian

            constexpr double pi = 3.14159265358979323846;
            double this_lon_x1 = std::cos(MinX / 180.0 * pi);
            double this_lon_y1 = std::sin(MinX / 180.0 * pi);
            double this_lon_x2 = std::cos(MaxX / 180.0 * pi);
            double this_lon_y2 = std::sin(MaxX / 180.0 * pi);

            double other_lon_x1 = std::cos(other.MinX / 180.0 * pi);
            double other_lon_y1 = std::sin(other.MinX / 180.0 * pi);
            double other_lon_x2 = std::cos(other.MaxX / 180.0 * pi);
            double other_lon_y2 = std::sin(other.MaxX / 180.0 * pi);

            double lon_cross_1 = this_lon_x1 * other_lon_y1 - this_lon_y1 * other_lon_x1;

            double lon_min_x, lon_min_y;
            double xy_min;
            if(lon_cross_1 >= 0) {
                minx_global = MinX;
                lon_min_x = this_lon_x1;
                lon_min_y = this_lon_y1;
            }
            else{
                minx_global = other.MinX;
                lon_min_x = other_lon_x1;
                lon_min_y = other_lon_y1;
            }

            double dot_1 = lon_min_x * this_lon_x2 + lon_min_y * this_lon_y2;
            double dot_2 = lon_min_x * other_lon_x2 + lon_min_y * other_lon_y2;

            if(dot_1 < dot_2){
                maxx_global = MaxX;
            }
            else{
                maxx_global = other.MaxX;
            }

            if(minx_global > maxx_global){
                maxx_global += 360.0;
            }

            MaxX = maxx_global;
            MinX = minx_global;

        }
    };
    /** Same as GDAL's OGRTriangle structure. See: https://gdal.org/doxygen/classOGRTriangle.html */
    typedef OGRTriangle Triangle;
}}
//end of file
