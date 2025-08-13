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
#include <iostream>

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

            // After this point, `X` mean longitude in degrees, and we are crossing the antimeridian.
            constexpr double pi = 3.14159265358979323846;

            // Compute unit vector from longitude in degrees
            auto deg_to_unitvec = [pi](double deg) {
                double rad = deg * pi / 180.0;
                std::cout << "deg_to_unitvec. deg=" << deg << ", vec=(" << std::cos(rad) << "," << std::sin(rad) << ")" << std::endl;
                return std::pair{std::cos(rad), std::sin(rad)};
            };

            // Compute dot product of two 2D vectors
            auto dot = [](const auto &a, const auto &b) {
                std::cout << "dot. a=(" << a.first << "," << a.second << "), b=(" << b.first << "," << b.second << "), result=" << (a.first * b.first + a.second * b.second) << std::endl;
                return a.first * b.first + a.second * b.second;

            };

            // Compute cross product (z-component only in 2D)
            auto cross = [](const auto &a, const auto &b) {
                std::cout << "cross. a=(" << a.first << "," << a.second << "), b=(" << b.first << "," << b.second << "), result=" << (a.first * b.second - a.second * b.first) << std::endl;
                return a.first * b.second - a.second * b.first;
            };

            // Compute the angle between two longitudes. Positive angle mean counter-clockwise direction.
            auto angle_between = [&](double from, double to){
                auto unitvec_from = deg_to_unitvec(from);
                auto unitvec_to = deg_to_unitvec(to);
                double angle = std::atan2(cross(unitvec_from, unitvec_to), dot(unitvec_from, unitvec_to)) * (180.0 / pi);
                angle += angle < 0 ? 360.0 : 0.0;
                std::cout << "angle_between. from=" << from << ", to=" << to << ", angle=" << angle << std::endl;
                return angle;
            };

            auto is_in_between = [&](double from, double to, double check){
                double tolerance = 1.0e-8;
                double angle_from_to = angle_between(from, to);
                double angle_from_check = angle_between(from, check);
                double angle_check_to = angle_between(check, to);
                std::cout << "is_in_between. from=" << from << ", to=" << to << ", check=" << check <<  ", result=" << (std::abs(angle_from_to - (angle_from_check + angle_check_to)) < tolerance) << std::endl;

                return (std::abs(angle_from_to - (angle_from_check + angle_check_to)) < tolerance);
            };

            // Check if this bbox contains the other bbox
            std::cout << "Checking the current bbox" << std::endl;
            if (is_in_between(MinX, MaxX, other.MinX) && is_in_between(MinX, MaxX, other.MaxX)) {
                // No need to do anything
                std::cout << "Keeping the current x bounds" << std::endl;
                return;
            }
            // Check if the other bbox contain this bbox
            std::cout << "Checking the other bbox" << std::endl;
            if (is_in_between(other.MinX, other.MaxX, MinX) && is_in_between(other.MinX, other.MaxX, MaxX)) {
                // Replace this bbox to others
                MinX = other.MinX;
                MaxX = other.MaxX;
                std::cout << "Replacing with the other x bounds" << std::endl;
                return;
            }

            // try merging other bbox into this bbox
            double global_xmin = MinX;
            double global_xmax = MaxX;
            double span_1 = 720.0; // A number sufficiently bigger than a cycle
            std::cout << "Checking the 1st candidate" << std::endl;
            if (is_in_between(MinX, other.MaxX, other.MinX)) {
                std::cout << "Adding [MinX, other.MaxX] into candidate" << std::endl;
                global_xmin = MinX;
                global_xmax = other.MaxX;
                span_1 = angle_between(global_xmin, global_xmax);
            }

            // try merging this bbox into other bbox
            std::cout << "Checking the 2nd candidate" << std::endl;
            if (is_in_between(other.MinX, MaxX, MinX)) {
                std::cout << "Adding [other.MinX, MaxX] into candidate" << std::endl;
                double span_2 = angle_between(other.MinX, MaxX);
                std::cout << "span_1=" << span_1 << ", span_2=" << span_2 << std::endl;
                if (span_1 > span_2) {
                    std::cout << "Choosing [other.MinX, MaxX]" << std::endl;
                    global_xmin = other.MinX;
                    global_xmax = MaxX;
                }
            }

            global_xmax += global_xmax < global_xmin ? 360.0 : 0.0;

            MinX = global_xmin;
            MaxX = global_xmax;

            // merge y boundary
            MinY = std::min(MinY, other.MinY);
            MaxY = std::max(MaxY, other.MaxY);

        }
    };
    /** Same as GDAL's OGRTriangle structure. See: https://gdal.org/doxygen/classOGRTriangle.html */
    typedef OGRTriangle Triangle;
}}
//end of file
