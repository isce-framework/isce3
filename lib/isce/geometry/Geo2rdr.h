//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_CORE_GEO2RDR_H
#define ISCE_CORE_GEO2RDR_H

// pyre
#include <portinfo>
#include <pyre/journal.h>

// isce::core
#include <isce/core/Metadata.h>
#include <isce/core/Orbit.h>
#include <isce/core/Poly2d.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Peg.h>
#include <isce/core/Pegtrans.h>
#include <isce/core/Projections.h>

// isce::io
#include <isce/io/Raster.h>

// isce::product
#include <isce/product/Product.h>

// Declaration
namespace isce {
    namespace geometry {
        class Geo2rdr;
    }
}

// Geo2rdr declaration
class isce::geometry::Geo2rdr {

    public:
        // Constructor from product
        inline Geo2rdr(isce::product::Product &);
        // Constructor from isce::core objects
        inline Geo2rdr(const isce::core::Ellipsoid &,
                       const isce::core::Orbit &,
                       const isce::core::Poly2d &,
                       const isce::core::Metadata &);

        // Set options
        inline void threshold(double);
        inline void numiter(int);
        inline void orbitMethod(isce::core::orbitInterpMethod);

        // Run geo2rdr - main entrypoint
        void geo2rdr(isce::io::Raster &,
                     const std::string &,
                     double, double);

        // Alternative: run geo2rdr with no constant offsets
        void geo2rdr(isce::io::Raster &,
                     const std::string &);

        // Value for null pixels
        const double NULL_VALUE = -1.0e6;

        // Getters for isce objects
        inline const isce::core::Orbit & orbit() const { return _orbit; }
        inline const isce::core::Ellipsoid & ellipsoid() const { return _ellipsoid; }
        inline const isce::core::Poly2d & doppler() const { return _doppler; }
        inline const isce::core::DateTime & sensingStart() const { return _sensingStart; }
        inline const isce::core::DateTime & refEpoch() const { return _refEpoch; }
        inline const isce::product::ImageMode & mode() const { return _mode; }

        // Get geo2rdr processing options
        inline double threshold() const { return _threshold; }
        inline int numiter() const { return _numiter; }
        inline isce::core::orbitInterpMethod orbitMethod() const { return _orbitMethod; }

    private:
        // Print extents and image info
        void _printExtents(pyre::journal::info_t &,
                           double, double, double,
                           double, double, double,
                           size_t, size_t);

        // Check we can interpolate orbit to middle of DEM
        void _checkOrbitInterpolation(double);

    private:
        // isce::core objects
        isce::core::Ellipsoid _ellipsoid;
        isce::core::Orbit _orbit;
        isce::core::Poly2d _doppler;
        isce::core::DateTime _refEpoch;
        isce::core::DateTime _sensingStart;

        // isce::product objects
        isce::product::ImageMode _mode;

        // Projection related data
        isce::core::ProjectionBase * _projTopo;

        // Processing parameters
        int _numiter;
        double _threshold;
        size_t _linesPerBlock = 1000;
        isce::core::orbitInterpMethod _orbitMethod;
};

// Get inline implementations for Geo2rdr
#define ISCE_GEOMETRY_GEO2RDR_ICC
#include "Geo2rdr.icc"
#undef ISCE_GEOMETRY_GEO2RDR_ICC

#endif

// end of file
