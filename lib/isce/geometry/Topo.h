//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_CORE_TOPO_H
#define ISCE_CORE_TOPO_H

// pyre
#include <portinfo>
#include <pyre/journal.h>

// isce::core
#include <isce/core/Peg.h>
#include <isce/core/Raster.h>

// isce::geometry
#include "geometry.h"
#include "TopoLayers.h"

// Declaration
namespace isce {
    namespace geometry {
        class Topo;
    }
}

// Declare Topo class
class isce::geometry::Topo {

    public:
        // Constructor: must have Ellipsoid, Orbit, and Metadata
        inline Topo(isce::core::Ellipsoid,
                    isce::core::Orbit,
                    isce::core::Metadata);

        // Set options
        inline void initialized(bool);
        inline void threshold(double);
        inline void numiter(int);
        inline void extraiter(int);
        inline void orbitMethod(isce::core::orbitInterpMethod);
        inline void demMethod(isce::core::dataInterpMethod);

        // Check initialization
        inline void checkInitialization(pyre::journal::info_t &) const;

        // Run topo - main entrypoint
        void topo(isce::core::Raster &,
                  isce::core::Poly2d &,
                  const std::string);

    private:

        // Get DEM bounds using first/last azimuth line and slant range bin
        void _computeDEMBounds(isce::core::Raster &,
                               DEMInterpolator &,
                               isce::core::Poly2d &);

        // Perform data initialization for a given azimuth line
        void _initAzimuthLine(int,
                              isce::core::StateVector &,
                              isce::core::Basis &);



        // Set output layers
        void _setOutputTopoLayers(cartesian_t &,
                                  TopoLayers &,
                                  isce::core::Pixel &,
                                  isce::core::StateVector &,
                                  isce::core::Basis &,
                                  DEMInterpolator &);

    private:
        // isce::core objects
        isce::core::Orbit _orbit;
        isce::core::Ellipsoid _ellipsoid;
        isce::core::Metadata _meta;
        isce::core::DateTime _refEpoch;
    
        // Optimization options
        double _threshold;
        int _numiter, _extraiter;
        isce::core::orbitInterpMethod _orbitMethod;
        isce::core::dataInterpMethod _demMethod;
        // Flag to make sure options have been initialized
        bool _initialized;
};

// Get inline implementations for Topo
#define ISCE_GEOMETRY_TOPO_ICC
#include "Topo.icc"
#undef ISCE_GEOMETRY_TOPO_ICC

const double MIN_H = -500.0;
const double MAX_H = -1000.0;
const double MARGIN = 0.15;

#endif

// end of file
