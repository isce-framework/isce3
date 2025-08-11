//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

#pragma once

#include <ogr_geometry.h>
#include <ogr_spatialref.h>
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

            OGRSpatialReference bbox_srs;
            bbox_srs.importFromEPSG(epsg);

            // Check if wrapping is necessary (i.e. geographic SRS AND crossing antimeridian).
            // If not, use the method in the base class
            if (!bbox_srs.IsGeographic() || ((maxx_global - minx_global) <= 180.0)) {
                OGREnvelope::Merge(other);
                return;
            }

            constexpr double pi = 3.14159265358979323846;

            auto lon_to_unitvec = [](double deg) {
                double rad = deg * pi / 180.0;
                return std::pair{ std::cos(rad), std::sin(rad) };
            };

            auto [this_lon_x1, this_lon_y1] = lon_to_unitvec(MinX);
            auto [this_lon_x2, this_lon_y2] = lon_to_unitvec(MaxX);
            auto [other_lon_x1, other_lon_y1] = lon_to_unitvec(other.MinX);
            auto [other_lon_x2, other_lon_y2] = lon_to_unitvec(other.MaxX);

            double lon_cross_1 = this_lon_x1 * other_lon_y1 - this_lon_y1 * other_lon_x1;

            double lon_min_x, lon_min_y;
            minx_global = (lon_cross_1 >= 0) ? MinX        : other.MinX;
            lon_min_x   = (lon_cross_1 >= 0) ? this_lon_x1 : other_lon_x1;
            lon_min_y   = (lon_cross_1 >= 0) ? this_lon_y1 : other_lon_y1;

            double dot_1 = lon_min_x * this_lon_x2 + lon_min_y * this_lon_y2;
            double dot_2 = lon_min_x * other_lon_x2 + lon_min_y * other_lon_y2;

            maxx_global = (dot_1 < dot_2) ? MaxX : other.MaxX;
            maxx_global += (minx_global > maxx_global) ? 360.0 : 0.0;

            MaxX = maxx_global;
            MinX = minx_global;

        }
    };
    /** Same as GDAL's OGRTriangle structure. See: https://gdal.org/doxygen/classOGRTriangle.html */
    typedef OGRTriangle Triangle;
}}
//end of file
