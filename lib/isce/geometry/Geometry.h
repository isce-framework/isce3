// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_GEOMETRY_H
#define ISCE_CORE_GEOMETRY_H

// std
#include <cmath>
#include <valarray>

// isce::core
#include <isce/core/Orbit.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Pegtrans.h>
#include <isce/core/StateVector.h>

// isce::geometry
#include "Basis.h"
#include "DemInterpolator.h"
#include "Pixel.h"

// Declaration
namespace isce {
    namespace geometry {
        class Geometry;
    }
}

// Geometry declaration
class isce::geometry::Geometry {

    // Public static methods
    public:
        // radar->geo
        static int rdr2geo(const Pixel &,
                           const Basis &,
                           const isce::core::StateVector &,
                           const isce::core::Ellipsoid &,
                           const isce::core::Pegtrans &,
                           const DEMInterpolator &,
                           cartesian_t &,
                           int, double, int, int);

        //static void geo2rdr(std::valarray<double> &, double, Poly2d &, double &, double &);
};

#endif

// end of file
