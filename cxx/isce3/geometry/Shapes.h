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

            // Check if antimeridian handling is necessary (i.e. geographic SRS AND crossing antimeridian).
            // If not, use the method in the base class
            if (!bbox_srs.IsGeographic() || ((maxx_global - minx_global) <= 180.0)) {
                OGREnvelope::Merge(other);
                return;
            }

            // Compute the angle between two longitudes. Always measure the angle in counter-clockwise.
            // The angle is defined in the range [0, 360).
            auto angle_between = [](double from, double to){
                double wrap_from = std::fmod(from + 360.0, 360.0);
                double wrap_to = std::fmod(to + 360.0, 360.0);

                double diff_angle = wrap_to - wrap_from;
                diff_angle += diff_angle < 0 ? 360.0 : 0.0;
                return diff_angle;
            };

            // Check if the angle between two longitudes is within a certain range
            auto is_in_between = [&](double from, double to, double check){
                double tolerance = 1.0e-8;
                double angle_from_to = angle_between(from, to);
                double angle_from_check = angle_between(from, check);
                double angle_check_to = angle_between(check, to);
                return (std::abs(angle_from_to - (angle_from_check + angle_check_to)) < tolerance);
            };

            // Check if this bbox contains the other bbox
            if (is_in_between(MinX, MaxX, other.MinX) && is_in_between(MinX, MaxX, other.MaxX)) {
                // No need to do anything
                return;
            }
            // Check if the other bbox contain this bbox
            if (is_in_between(other.MinX, other.MaxX, MinX) && is_in_between(other.MinX, other.MaxX, MaxX)) {
                // Replace this bbox to others
                MinX = other.MinX;
                MaxX = other.MaxX;
                return;
            }

            // try merging other bbox into this bbox
            minx_global = MinX;
            maxx_global = MaxX;
            double span_1 = 720.0; // A number sufficiently bigger than a cycle
            if (is_in_between(MinX, other.MaxX, other.MinX)) {
                minx_global = MinX;
                maxx_global = other.MaxX;
                span_1 = angle_between(minx_global, maxx_global);
            }

            // try merging this bbox into other bbox
            if (is_in_between(other.MinX, MaxX, MinX)) {
                double span_2 = angle_between(other.MinX, MaxX);
                if (span_1 > span_2) {
                    minx_global = other.MinX;
                    maxx_global = MaxX;
                }
            }

            // Wrap the 'east' part of the bbox boundary if necessary
            maxx_global += maxx_global < minx_global ? 360.0 : 0.0;

            MinX = minx_global;
            MaxX = maxx_global;

            // merge y boundary
            MinY = std::min(MinY, other.MinY);
            MaxY = std::max(MaxY, other.MaxY);

        }
    };
    /** Same as GDAL's OGRTriangle structure. See: https://gdal.org/doxygen/classOGRTriangle.html */
    typedef OGRTriangle Triangle;
}}
//end of file
