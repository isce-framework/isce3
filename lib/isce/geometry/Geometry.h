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
#include "Constants.h"
#include "Orbit.h"
#include "Ellipsoid.h"
#include "Interpolator.h"
#include "Raster.h"

// Declaration
namespace isce {
    namespace core {
        class Geometry;
    }
}

// Geometry declaration
class isce::core::Geometry {

    // Public static methods
    public:
        // radar->geo when 
        //static void rdr2geo(double, double, Orbit &, Ellipsoid &, std::valarray<double> &,
        //                    int numIter=10);
        // geo->rdr
        static void geo2rdr(std::valarray<double> &, double, Poly2d &, double &, double &);
};

#endif

// end of file
