//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel, Joshua Cohen
// Copyright 2017-2018

#pragma once

// pyre
#include <pyre/journal.h>

// isce3::core
#include <isce3/core/Metadata.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Poly2d.h>
#include <isce3/core/LUT1d.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Peg.h>
#include <isce3/core/Projections.h>

// isce3::io
#include <isce3/io/Raster.h>

// isce3::product
#include <isce3/product/Product.h>
#include <isce3/product/RadarGridParameters.h>

#include <limits>

// Declaration
namespace isce3 {
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
class isce3::geometry::Geo2rdr {
public:

    /**
     * Constructor from product
     *
     * @param[in] product Input Product
     * @param[in] frequency Frequency designation
     * @param[in] nativeDoppler Flag for using native Doppler frequencies instead of zero-Doppler
     */
    Geo2rdr(const isce3::product::Product &,
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
    Geo2rdr(const isce3::core::Ellipsoid &,
            const isce3::core::Orbit &,
            const isce3::core::LUT2d<double> &,
            const isce3::core::Metadata &);

    /**
     * Constructor from core objects
     *
     * @param[in] radarGrid RadarGridParameters object
     * @param[in] orbit     Orbit object
     * @param[in] ellipsoid Ellipsoid object
     * @param[in] doppler   LUT2d doppler model
     */
    Geo2rdr(const isce3::product::RadarGridParameters & radarGrid,
            const isce3::core::Orbit & orbit,
            const isce3::core::Ellipsoid & ellipsoid,
            const isce3::core::LUT2d<double> & doppler = {});

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
     * Set lines to be processed per block
     *
     * @param[in] linesPerBlock Lines to be processed per block
     */
    void linesPerBlock(size_t linesPerBlock) { _linesPerBlock = linesPerBlock; }

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
    void geo2rdr(isce3::io::Raster & topoRaster,
                 isce3::io::Raster & rgoffRaster,
                 isce3::io::Raster & azoffRaster,
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
    void geo2rdr(isce3::io::Raster & topoRaster,
                 const std::string & outdir,
                 double azshift=0.0, double rgshift=0.0);

    /** NoData Value*/
    const double NULL_VALUE =  -1000000.0;

    // Getters for isce objects

    /** Get Orbit object used for processing */
    const isce3::core::Orbit & orbit() const { return _orbit; }

    /** Get Ellipsoid object used for processing */
    const isce3::core::Ellipsoid & ellipsoid() const { return _ellipsoid; }

    /** Get Doppler model used for processing */
    const isce3::core::LUT2d<double> & doppler() const { return _doppler; }

    /** Get read-only reference to RadarGridParameters */
    const isce3::product::RadarGridParameters & radarGridParameters() const { return _radarGrid; }

    // Get geo2rdr processing options

    /** Return the azimuth time convergence threshold used for processing */
    double threshold() const { return _threshold; }

    /** Return number of Newton-Raphson iterations used for processing */
    int numiter() const { return _numiter; }

    /** Get linesPerBlock */
    size_t linesPerBlock() const { return _linesPerBlock; }

private:

    /** Print information for debugging */
    void _printExtents(pyre::journal::info_t &,
                       double, double, double,
                       double, double, double,
                       size_t, size_t);

    /** Quick check to ensure we can interpolate orbit to middle of DEM*/
    void _checkOrbitInterpolation(double);

    // isce3::core objects
    isce3::core::Ellipsoid _ellipsoid;
    isce3::core::Orbit _orbit;
    isce3::core::LUT2d<double> _doppler;

    // RadarGridParameters
    isce3::product::RadarGridParameters _radarGrid;

    // Projection related data
    isce3::core::ProjectionBase * _projTopo;

    // Processing parameters
    int _numiter;
    double _threshold = 1e-8;
    size_t _linesPerBlock = 1000;
};

// Get inline implementations for Geo2rdr
#define ISCE_GEOMETRY_GEO2RDR_ICC
#include "Geo2rdr.icc"
#undef ISCE_GEOMETRY_GEO2RDR_ICC
