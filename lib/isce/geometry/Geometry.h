// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_GEOMETRY_H
#define ISCE_CORE_GEOMETRY_H

// std
#include <valarray>

// isce::core
#include "isce/core/Constants.h"
#include "isce/core/Orbit.h"
#include "isce/core/Ellipsoid.h"
#include "isce/core/Interpolator.h"
#include "isce/core/Raster.h"

// Declaration
namespace isce {
    namespace geometry {
        // Expose some useful isce::core typedefs
        typedef isce::core::cartesian_t cartesian_t;
        class Geometry;
    }
}

// Geometry declaration
class isce::geometry::Geometry {

    // Public static methods
    public:
        // radar->geo when 
        static int rdr2geo(double, double, isce::core::Orbit &, isce::core::Ellipsoid &,
                           std::valarray<double> &, int numIter=10);
        static void geo2rdr(std::valarray<double> &, double, Poly2d &, double &, double &);
};

#endif

// end of file
