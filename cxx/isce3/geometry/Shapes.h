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
        // Expose the OGREnvelope::Merge method, so that the old function interface is preserved.
        using OGREnvelope::Merge;

        //Overload the method by adding parameter for EPSG in geographic coordinates.
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

            // After this point, `X` mean longitude in degrees, and we are crossing the antimeridian.
            constexpr double pi = 3.14159265358979323846;

            // Compute unit vector from longitude in degrees
            auto deg_to_unitvec = [](double deg) {
                double rad = deg * pi / 180.0;
                return std::pair{ std::cos(rad), std::sin(rad) };
            };

            // Compute dot product of two 2D vectors
            auto dot = [](auto a, auto b) {
                return a.first * b.first + a.second * b.second;
            };

            // Compute cross product (z-component only in 2D)
            auto cross = [](auto a, auto b) {
                return a.first * b.second - a.second * b.first;
            };

            // compute the unit vectors for the min / max longitudes of both bounding boxes
            auto unitvec_this_minx = deg_to_unitvec(MinX);
            auto unitvec_this_maxx = deg_to_unitvec(MaxX);
            auto unitvec_other_minx = deg_to_unitvec(other.MinX);
            auto unitvec_other_maxx = deg_to_unitvec(other.MaxX);

            // compute the cross product of the min longitude unit vectors,
            // which tells us which bounding box is "to the left" of the other
            double lon_cross_1 = cross(unitvec_this_minx, unitvec_other_minx);

            // Determine which bounding box has to be placed in the left.
            auto unitvec_min_lon = (lon_cross_1 >= 0) ?
                                   unitvec_this_minx :
                                   unitvec_other_minx;

            double dot_1 = dot(unitvec_min_lon, unitvec_this_maxx);
            double dot_2 = dot(unitvec_min_lon, unitvec_other_maxx);

            minx_global = (lon_cross_1 >= 0) ? MinX : other.MinX;
            maxx_global = (dot_1 < dot_2) ? MaxX : other.MaxX;
            maxx_global += (minx_global > maxx_global) ? 360.0 : 0.0;

            MaxX = maxx_global;
            MinX = minx_global;

            // Merge the Y coordinates
            MinY = std::min(MinY, other.MinY);
            MaxY = std::max(MaxY, other.MaxY);
        }
    };
    /** Same as GDAL's OGRTriangle structure. See: https://gdal.org/doxygen/classOGRTriangle.html */
    typedef OGRTriangle Triangle;
}}
//end of file
