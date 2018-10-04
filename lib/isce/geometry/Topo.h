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
/** Transformer from radar geometry coordinates to map coordinates with DEM / reference altitude
 *
 * See <a href="overview_geometry.html#forwardgeom">geometry overview</a> for a description of the algorithm*/
class isce::geometry::Topo {

    public:
        /** Constructor using a product*/
        inline Topo(isce::product::Product &);
        /** Constructor using core objects*/
        inline Topo(isce::core::Ellipsoid,
                    isce::core::Orbit,
                    isce::core::Poly2d,
                    isce::core::Metadata);
        
        /** Set initialization flag*/
        inline void initialized(bool);
        /** Set convergence threshold */
        inline void threshold(double);
        /** Set number of primary iterations */
        inline void numiter(int);
        /** Set number of secondary iterations */
        inline void extraiter(int);
        /** Set orbit interpolation method */
        inline void orbitMethod(isce::core::orbitInterpMethod);
        /** Set DEM interpolation method */
        inline void demMethod(isce::core::dataInterpMethod);
        /** Set output coordinate system */
        inline void epsgOut(int);

        /** Check initialization of processing module */
        inline void checkInitialization(pyre::journal::info_t &) const;

        /** Main entry point for the module */
        void topo(isce::io::Raster &, const std::string);

    private:

        /** Compute the DEM bounds for given product */ 
        void _computeDEMBounds(isce::io::Raster &,
                               DEMInterpolator &,
                               size_t, size_t);

        /** Initialize TCN basis for given azimuth line */
        void _initAzimuthLine(size_t,
                              isce::core::StateVector &,
                              isce::core::Basis &);

        /** Write to output layers */
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
