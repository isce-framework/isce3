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

        // Overload the method by adding parameter for EPSG in geographic coordinates.
        // If the bounding boxes are in geographic coordinates, the overall longitude
        // extent is assumed to be <= 180 degrees. If an antimeridian crossing is
        // detected, the longitude coordinates are re-wrapped to the interval [0, 360).
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

            // Wrap the input angle (in degrees) to the interval [0, 360).
            auto wrap360 = [](double x) {
                auto y = std::fmod(x, 360.0);
                if (y < 0.0) {
                    y += 360.0;
                }
                return y;
            };

            double MinX_wrap = wrap360(MinX);
            double MaxX_wrap = wrap360(MaxX);

            double other_MinX_wrap = wrap360(other.MinX);
            double other_MaxX_wrap = wrap360(other.MaxX);


            // In general, `MinX` is the east edge and `MaxX` is the west edge,
            // unless the bounding box contained the antimeridian and the
            // coordinates weren't "unwrapped" (such as if the bounding box
            // was created using `Perimeter::getEnvelope()`). We assume this is
            // the case if the difference between `MaxX` and `MinX` exceeds 180
            // degrees. In this case, we can normalize the bounding box by
            // swapping `MinX` and `MaxX` and re-wrapping the coordinates to the
            // interval [0, 360).
            if (MaxX - MinX > 180.0) {
                std::swap(MinX_wrap, MaxX_wrap);
            }
            if (other.MaxX - other.MinX > 180.0) {
                std::swap(other_MinX_wrap, other_MaxX_wrap);
            }

            MinX = std::min(MinX_wrap, other_MinX_wrap);
            MaxX = std::max(MaxX_wrap, other_MaxX_wrap);

            // merge y boundary
            MinY = std::min(MinY, other.MinY);
            MaxY = std::max(MaxY, other.MaxY);

        }
    };
    /** Same as GDAL's OGRTriangle structure. See: https://gdal.org/doxygen/classOGRTriangle.html */
    typedef OGRTriangle Triangle;
}}
//end of file
