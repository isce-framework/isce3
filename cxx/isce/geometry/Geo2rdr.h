//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel, Joshua Cohen
// Copyright 2017-2018

#pragma once

// pyre
#include <pyre/journal.h>

// isce::core
#include <isce/core/Metadata.h>
#include <isce/core/Orbit.h>
#include <isce/core/Poly2d.h>
#include <isce/core/LUT1d.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Peg.h>
#include <isce/core/Projections.h>

// isce::io
#include <isce/io/Raster.h>

// isce::product
#include <isce/product/Product.h>
#include <isce/product/RadarGridParameters.h>

// Declaration
namespace isce {
    namespace geometry {
        class Geo2rdr;
    }
}

/**
 * Transformer from map coordinates to radar geometry coordinates.
 *
 * See <a href="overview_geometry.html#inversegeom">geometry overview</a>
 * for description of the algorithm.
 */
class isce::geometry::Geo2rdr {
public:

    /**
     * Constructor from product
     *
     * @param[in] product Input Product
     * @param[in] frequency Frequency designation
     * @param[in] nativeDoppler Flag for using native Doppler frequencies instead of zero-Doppler
     */
    Geo2rdr(const isce::product::Product &,
            char frequency = 'A',
            bool nativeDoppler = false);

    /**
     * Constructor from core objects
     *
     * @param[in] ellipsoid Ellipsoid object
     * @param[in] orbit Orbit object
     * @param[in] doppler LUT1d doppler model
     * @param[in] meta Metadata object
     */
    Geo2rdr(const isce::core::Ellipsoid &,
            const isce::core::Orbit &,
            const isce::core::LUT2d<double> &,
            const isce::core::Metadata &);

    /**
     * Constructor from core objects
     *
     * @param[in] radarGrid RadarGridParameters object
     * @param[in] orbit     Orbit object
     * @param[in] ellipsoid Ellipsoid object
     * @param[in] doppler   LUT2d doppler model
     */
    Geo2rdr(const isce::product::RadarGridParameters & radarGrid,
            const isce::core::Orbit & orbit,
            const isce::core::Ellipsoid & ellipsoid,
            const isce::core::LUT2d<double> & doppler = {});

    /**
     * Set convergence threshold
     *
     * @param[in] t Azimuth time convergence threshold in seconds
     */
    void threshold(double t) { _threshold = t; }

    /**
     * Set number of Newton-Raphson iterations
     *
     * @param[in] n Max number of Newton-Raphson iterations
     */
    void numiter(int n) { _numiter = n; }

    /**
     * Run geo2rdr with offsets and externally created offset rasters
     *
     * @param[in] topoRaster outputs of topo - i.e, pixel-by-pixel x,y,h as bands
     * @param[in] outdir directory to write outputs to
     * @param[in] rgoffRaster range offset output
     * @param[in] azoffRaster azimuth offset output
     * @param[in] azshift Number of lines to shift by in azimuth
     * @param[in] rgshift Number of pixels to shift by in range
     */
    void geo2rdr(isce::io::Raster & topoRaster,
                 isce::io::Raster & rgoffRaster,
                 isce::io::Raster & azoffRaster,
                 double azshift=0.0, double rgshift=0.0);

    /**
     * Run geo2rdr with constant offsets and internally created offset rasters
     *
     * This is the main geo2rdr driver. The pixel-by-pixel output filenames are fixed for now
     * <ul>
     * <li>azimuth.off - Azimuth offset to be applied to product to align with topoRaster
     * <li>range.off - Range offset to be applied to product to align with topoRaster
     *
     * @param[in] topoRaster outputs of topo -i.e, pixel-by-pixel x,y,h as bands
     * @param[in] outdir directory to write outputs to
     * @param[in] azshift Number of lines to shift by in azimuth
     * @param[in] rgshift Number of pixels to shift by in range
     */
    void geo2rdr(isce::io::Raster & topoRaster,
                 const std::string & outdir,
                 double azshift=0.0, double rgshift=0.0);

    /** NoData Value*/
    const double NULL_VALUE = -1.0e6;

    // Getters for isce objects

    /** Get Orbit object used for processing */
    const isce::core::Orbit & orbit() const { return _orbit; }

    /** Get Ellipsoid object used for processing */
    const isce::core::Ellipsoid & ellipsoid() const { return _ellipsoid; }

    /** Get Doppler model used for processing */
    const isce::core::LUT2d<double> & doppler() const { return _doppler; }

    /** Get read-only reference to RadarGridParameters */
    const isce::product::RadarGridParameters & radarGridParameters() const { return _radarGrid; }

    // Get geo2rdr processing options

    /** Return the azimuth time convergence threshold used for processing */
    double threshold() const { return _threshold; }

    /** Return number of Newton-Raphson iterations used for processing */
    int numiter() const { return _numiter; }

private:

    /** Print information for debugging */
    void _printExtents(pyre::journal::info_t &,
                       double, double, double,
                       double, double, double,
                       size_t, size_t);

    /** Quick check to ensure we can interpolate orbit to middle of DEM*/
    void _checkOrbitInterpolation(double);

    // isce::core objects
    isce::core::Ellipsoid _ellipsoid;
    isce::core::Orbit _orbit;
    isce::core::LUT2d<double> _doppler;

    // RadarGridParameters
    isce::product::RadarGridParameters _radarGrid;

    // Projection related data
    isce::core::ProjectionBase * _projTopo;

    // Processing parameters
    int _numiter;
    double _threshold;
    size_t _linesPerBlock = 1000;
};

// Get inline implementations for Geo2rdr
#define ISCE_GEOMETRY_GEO2RDR_ICC
#include "Geo2rdr.icc"
#undef ISCE_GEOMETRY_GEO2RDR_ICC
