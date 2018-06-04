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

// isce::io
#include <isce/io/Raster.h>

// Declaration
namespace isce {
    namespace geometry {
        class Geo2rdr;
    }
}

// Geo2rdr declaration
class isce::geometry::Geo2rdr {

    public:
        // Constructor: must have Ellipsoid, Orbit, and Metadata
        Geo2rdr(isce::core::Ellipsoid,
                isce::core::Orbit,
                isce::core::Metadata);

        // Set options
        inline void threshold(double);
        inline void numiter(int);
        inline void orbitMethod(isce::core::orbitInterpMethod);

        // Run geo2rdr - main entrypoint
        void geo2rdr(isce::io::Raster &,
                     isce::io::Raster &,
                     isce::io::Raster &,
                     isce::core::Poly2d &,
                     const std::string &,
                     double, double);

        // Alternative: run geo2rdr with no constant offsets
        void geo2rdr(isce::io::Raster &,
                     isce::io::Raster &,
                     isce::io::Raster &,
                     isce::core::Poly2d &,
                     const std::string &);

        // Value for null pixels
        const double NULL_VALUE = -1000.0;

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
        isce::core::Metadata _meta;
        isce::core::DateTime _refEpoch;

        // Processing parameters
        int _numiter;
        double _threshold;
        isce::core::orbitInterpMethod _orbitMethod;
};

// Get inline implementations for Geo2rdr
#define ISCE_GEOMETRY_GEO2RDR_ICC
#include "Geo2rdr.icc"
#undef ISCE_GEOMETRY_GEO2RDR_ICC

#endif

// end of file
