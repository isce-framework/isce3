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
            double maxx_global = std:max(MaxX, other.MaxX);

            // Check if wrapping is necessary. If not, use the method in the base class
            if ((epsg != 4326) || (maxx_global - minx_global) <= 180.0){
                // just use the method in the base class
                OGREnvelope::Merge(other);
                return;
            }

            // Check if wrapping is really necessary. Wrap is necessary.
            if isAntimeridianCrossing() || other.isAntimeridianCrossing()) {
            // Wrap the angles to the range [0, 360) if either of the
            // bounding boxes crosses the antimeridian
            auto wrap = [](double angle) {
                return std::fmod(std::fmod(angle, 360.0) + 360.0, 360.0);
            };

                MinX = std::min(wrap(MinX), wrap(other.MinX));
                MaxX = std::max(wrap(MaxX), wrap(other.MaxX));
                MinY = std::min(MinY, other.MinY);
                MaxY = std::max(MaxY, other.MaxY);
            }
        };
    /** Same as GDAL's OGRTriangle structure. See: https://gdal.org/doxygen/classOGRTriangle.html */
    typedef OGRTriangle Triangle;
}}
//end of file
