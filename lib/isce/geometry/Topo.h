//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_GEOMETRY_TOPO_H
#define ISCE_GEOMETRY_TOPO_H

// pyre
#include <pyre/journal.h>

// isce::core
#include <isce/core/Metadata.h>
#include <isce/core/Peg.h>

// isce::io
#include <isce/io/Raster.h>

// isce::product
#include <isce/product/Product.h>
#include <isce/product/RadarGridParameters.h>

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
        inline Topo(const isce::product::Product &,
                    char frequency = 'A',
                    bool nativeDoppler = false,
                    size_t numberAzimuthLooks = 1,
                    size_t numberRangeLooks = 1);

        /** Constructor using core objects*/
        inline Topo(const isce::core::Ellipsoid &,
                    const isce::core::Orbit &,
                    const isce::core::LUT2d<double> &,
                    const isce::core::Metadata &,
                    size_t numberAzimuthLooks = 1,
                    size_t numberRangeLooks = 1);
        
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
        /** Set mask computation flag */
        inline void computeMask(bool);

        // Get topo processing options
        /** Get lookSide used for processing */
        inline int lookSide() const { return _lookSide; }
        /** Get distance convergence threshold used for processing */
        inline double threshold() const { return _threshold; }
        /** Get number of primary iterations used for processing */
        inline int numiter() const { return _numiter; }
        /** Get number of secondary iterations used for processing*/
        inline int extraiter() const { return _extraiter; }
        /** Get the output coordinate system used for processing */
        inline int epsgOut() const { return _epsgOut; }
        /** Get the DEM interpolation method used for processing */
        inline isce::core::dataInterpMethod demMethod() const { return _demMethod; }
        /** Get mask computation flag */
        inline bool computeMask() const { return _computeMask; }

        /** Get read-only reference to RadarGridParameters */
        inline const isce::product::RadarGridParameters & radarGridParameters() const {
            return _radarGridParameters;
        }

        // Get DEM bounds using first/last azimuth line and slant range bin
        void computeDEMBounds(isce::io::Raster &,
                              DEMInterpolator &,
                              size_t, size_t);

        /** Main entry point for the module; internal creation of topo rasters */
        void topo(isce::io::Raster &, const std::string);

        /** Run topo with externally created topo rasters in TopoLayers object */
        void topo(isce::io::Raster & demRaster, TopoLayers & layers);

        /** Run topo with externally created topo rasters; generate mask */
        void topo(isce::io::Raster & demRaster, isce::io::Raster & xRaster,
                  isce::io::Raster & yRaster, isce::io::Raster & heightRaster,
                  isce::io::Raster & incRaster, isce::io::Raster & hdgRaster,
                  isce::io::Raster & localIncRaster, isce::io::Raster & localPsiRaster,
                  isce::io::Raster & simRaster, isce::io::Raster & maskRaster);

        /** Run topo with externally created topo rasters; generate mask */
        void topo(isce::io::Raster & demRaster, isce::io::Raster & xRaster,
                  isce::io::Raster & yRaster, isce::io::Raster & heightRaster,
                  isce::io::Raster & incRaster, isce::io::Raster & hdgRaster,
                  isce::io::Raster & localIncRaster, isce::io::Raster & localPsiRaster,
                  isce::io::Raster & simRaster);

        /** Compute layover/shadow masks */
        void setLayoverShadow(TopoLayers &,
                              DEMInterpolator &,
                              std::vector<isce::core::cartesian_t> &);

        // Getters for isce objects
        /** Get the orbits used for processing */
        inline const isce::core::Orbit & orbit() const { return _orbit; }
        /** Get the ellipsoid used for processing */
        inline const isce::core::Ellipsoid & ellipsoid() const { return _ellipsoid; }
        /** Get the doppler module used for processing */
        inline const isce::core::LUT2d<double> & doppler() const { return _doppler; }

    private:

        /** Initialize TCN basis for given azimuth line */
        void _initAzimuthLine(size_t,
                              double &,
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
        isce::core::LUT2d<double> _doppler;

        // RadarGridParameters
        isce::product::RadarGridParameters _radarGridParameters;
        
        // Optimization options
        double _threshold = 1.0e-8;
        int _numiter = 25;
        int _extraiter = 10;
        int _lookSide;
        size_t _linesPerBlock = 1000;
        bool _computeMask = true;
        isce::core::orbitInterpMethod _orbitMethod;
        isce::core::dataInterpMethod _demMethod;

        // Output options and objects
        int _epsgOut;
        isce::core::ProjectionBase * _proj;
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
