//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

#pragma once

//isce::product
#include <isce/product/RadarGridParameters.h>

//isce::geometry
#include "Shapes.h"
#include "DEMInterpolator.h"

//Declaration
namespace isce{
    namespace geometry{

    /** Compute the perimeter of a radar grid in map coordinates.
    * 
    * @param[in] radarGrid    RadarGridParameters object
    * @param[in] orbit         Orbit object
    * @param[in] proj          ProjectionBase object indicating desired projection of output. 
    * @param[in] doppler       LUT2d doppler model
    * @param[in] demInterp     DEM Interpolator
    * @param[in] pointsPerEge  Number of points to use on each edge of radar grid
    * @param[in] threshold     Slant range threshold for convergence 
    * @param[in] numiter       Max number of iterations for convergence
    *
    * The outputs of this method is an OGRLinearRing.
    * Transformer from radar geometry coordinates to map coordiantes with a DEM
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

    /** Compute bounding box using min/ max altitude for quick estimates
    *
    * @param[in] radarGrid    RadarGridParameters object
    * @param[in] orbit         Orbit object
    * @param[in] proj          ProjectionBase object indicating desired projection of output. 
    * @param[in] doppler       LUT2d doppler model
    * @param[in] hgts          Vector of heights to use for the scene
    * @param[in] margin        Marging to add to estimated bounding box in decimal degrees
    * @param[in] pointsPerEge  Number of points to use on each edge of radar grid
    * @param[in] threshold     Slant range threshold for convergence 
    * @param[in] numiter       Max number of iterations for convergence
    *
    * The output of this method is an OGREnvelope.
    */
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
