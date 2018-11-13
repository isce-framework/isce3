// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//
/** \file geometry.h
 * Collection of simple commonly used geometry functions
 *
 * There are no classes defined in this file. Its a collection of functions 
 * that are meant to be light weight versions of isce::geometry::Topo and 
 * isce::geometry::Geo2rdr.*/
#ifndef ISCE_CORE_GEOMETRY_H
#define ISCE_CORE_GEOMETRY_H

// std
#include <cmath>
#include <valarray>

// isce::core
#include <isce/core/Constants.h>
#include <isce/core/Basis.h>
#include <isce/core/Orbit.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Metadata.h>
#include <isce/core/Pegtrans.h>
#include <isce/core/Pixel.h>
#include <isce/core/Poly2d.h>
#include <isce/core/StateVector.h>

// isce::product
#include <isce/product/ImageMode.h>

// isce::geometry
#include "DEMInterpolator.h"

// Declaration
namespace isce {
    namespace geometry {

        // Some useful typedefs from isce::core
        typedef isce::core::cartesian_t cartesian_t;
        typedef isce::core::cartmat_t cartmat_t;

        /** Radar geometry coordinates to map coordinates transformer*/
        int rdr2geo(double, double, double,
                    const isce::core::Orbit &,
                    const isce::core::Ellipsoid &,
                    const DEMInterpolator &,
                    cartesian_t &,
                    double, int, double, int, int,
                    isce::core::orbitInterpMethod); 
       
        /** Radar geometry coordinates to map coordinates transformer*/
        int rdr2geo(const isce::core::Pixel &,
                    const isce::core::Basis &,
                    const isce::core::StateVector &,
                    const isce::core::Ellipsoid &,
                    const DEMInterpolator &,
                    cartesian_t &,
                    int, double, int, int);

        /** Map coordinates to radar geometry coordinates transformer*/
        int geo2rdr(const cartesian_t &,
                    const isce::core::Ellipsoid &,
                    const isce::core::Orbit &,
                    const isce::core::Poly2d &,
                    const isce::product::ImageMode &,
                    double &, double &,
                    double, int, double);

        // Utility function to compute geocentric TCN basis from state vector
        void geocentricTCN(isce::core::StateVector &,
                           isce::core::Basis &);    

    }
}

#endif

// end of file
