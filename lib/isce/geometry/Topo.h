//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_GEOMETRY_TOPO_H
#define ISCE_GEOMETRY_TOPO_H

// pyre
#include <portinfo>
#include <pyre/journal.h>

// isce::core
#include <isce/core/Metadata.h>
#include <isce/core/Peg.h>

// isce::io
#include <isce/io/Raster.h>

// isce::product
#include <isce/product/Product.h>

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
        // Constructor from Product
        inline Topo(isce::product::Product &);
        // Constructor from isce::core objects
        inline Topo(isce::core::Ellipsoid,
                    isce::core::Orbit,
                    isce::core::Poly2d,
                    isce::core::Metadata);
        
        // Set topo processing options
        inline void initialized(bool);
        inline void threshold(double);
        inline void numiter(int);
        inline void extraiter(int);
        inline void orbitMethod(isce::core::orbitInterpMethod);
        inline void demMethod(isce::core::dataInterpMethod);
        inline void epsgOut(int);

        // Get topo processing options
        inline int lookSide() const { return _lookSide; }
        inline double threshold() const { return _threshold; }
        inline int numiter() const { return _numiter; }
        inline int extraiter() const { return _extraiter; }

        // Check initialization
        inline void checkInitialization(pyre::journal::info_t &) const;

        // Get DEM bounds using first/last azimuth line and slant range bin
        void computeDEMBounds(isce::io::Raster &,
                              DEMInterpolator &,
                              size_t, size_t);

        // Run topo - main entrypoint
        void topo(isce::io::Raster &, const std::string);

        // Getters for isce objects
        inline const isce::core::Orbit & orbit() const { return _orbit; }
        inline const isce::core::Ellipsoid & ellipsoid() const { return _ellipsoid; }
        inline const isce::core::Poly2d & doppler() const { return _doppler; }
        inline const isce::core::DateTime & sensingStart() const { return _sensingStart; }
        inline const isce::product::ImageMode & mode() const { return _mode; }

    private:

        // Perform data initialization for a given azimuth line
        void _initAzimuthLine(size_t,
                              isce::core::StateVector &,
                              isce::core::Basis &);

        // Set output layers
        void _setOutputTopoLayers(cartesian_t &,
                                  TopoLayers &,
                                  size_t,
                                  isce::core::Pixel &,
                                  isce::core::StateVector &,
                                  isce::core::Basis &,
                                  DEMInterpolator &);

    private:
        // isce::core objects
        isce::core::Orbit _orbit;
        isce::core::Ellipsoid _ellipsoid;
        isce::core::Poly2d _doppler;
        isce::core::DateTime _sensingStart, _refEpoch;

        // isce::product objects
        isce::product::ImageMode _mode;
    
        // Optimization options
        double _threshold;
        int _numiter, _extraiter;
        int _lookSide;
        size_t _linesPerBlock = 1000;
        isce::core::orbitInterpMethod _orbitMethod;
        isce::core::dataInterpMethod _demMethod;

        // Output options and objects
        int _epsgOut;
        isce::core::ProjectionBase * _proj;

        // Flag to make sure options have been initialized
        bool _initialized;
};

// Get inline implementations for Topo
#define ISCE_GEOMETRY_TOPO_ICC
#include "Topo.icc"
#undef ISCE_GEOMETRY_TOPO_ICC

const double MIN_H = -500.0;
const double MAX_H = -1000.0;
const double MARGIN = 0.15 * M_PI / 180.0;

#endif

// end of file
