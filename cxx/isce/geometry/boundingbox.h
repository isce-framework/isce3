//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

#pragma once

//isce::core
#include <isce/core/Ellipsoid.h>

//isce::product
#include <isce/product/RadarGridParameters.h>

//isce::geometry
#include "geometry.h"
#include "Shapes.h"
#include "DEMInterpolator.h"

//Declaration
namespace isce{
    namespace geometry{

    /** Transformer from radar geometry coordinates to map coordiantes with a DEM
    * The sequence of walking the perimeter is always in the following order :
    * <ul>
    * <li> Start at Early Time, Near Range edge. Always the first point of the polygon.
    * <li> From there, Walk along the Early Time edge to Early Time, Far Range.
    * <li> From there, walk along the Far Range edge to Late Time, Far Range.
    * <li> From there, walk along the Late Time edge to Late Time, Near Range. 
    * <li> From there, walk along the Near Range edge back to Early Time, Near Range.
    * </ul>
    */
    Perimeter
    getGeoPerimeter(const isce::product::RadarGridParameters &radarGrid,
                const isce::core::Orbit &orbit,
                const isce::core::ProjectionBase *proj,
                const isce::core::LUT2d<double> &doppler = {},
                const DEMInterpolator& demInterp = DEMInterpolator(0.),
                const int pointsPerEdge = 11,
                const double threshold = 1.0e-8,
                const int numiter = 15);

    /** Compute bounding box using min/ max altitude for quick estimates*/
    BoundingBox
    getGeoBoundingBox(const isce::product::RadarGridParameters &radarGrid,
                      const isce::core::Orbit &orbit,
                      const isce::core::ProjectionBase *proj,
                      const isce::core::LUT2d<double> &doppler = {},
                      const std::vector<double> &hgts = {isce::core::GLOBAL_MIN_HEIGHT, isce::core::GLOBAL_MAX_HEIGHT},
                      const double margin = 0.0,
                      const int pointsPerEdge = 11,
                      const double threshold = 1.0e-8,
                      const int numiter = 15);


}
}
//end of file
