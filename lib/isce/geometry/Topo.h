//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan Riel
// Copyright 2017-2018

#ifndef ISCE_CORE_TOPO_H
#define ISCE_CORE_TOPO_H

// pyre
#include <pyre/journal.h>

// isce::core
#include "Metadata.h"
#include "Orbit.h"
#include "Poly2d.h"
#include "Ellipsoid.h"
#include "Raster.h"

// Declaration
namespace isce {
    namespace core {
        class Topo;
    }
}

// Declare Topo class
class isce::core::Topo {

    public:
        // Constructors
        Topo();
        Topo(Ellipsoid &, Orbit &);

        // Run topo
        void topo(Raster &, Poly2d &, Poly2d &, const std::string);

    private:
        // isce::core objects
        Orbit & _orbit;
        Ellipsoid & _ellipsoid;
        Metadata & _meta;
    
        // Optimization options
        int _numiter, _extraiter, _orbit_method, _dem_method;

        // Main radar->geo iterations
        _rdr2geo(double, double);
};

#endif

// end of file
