//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel, Joshua Cohen
// Copyright 2017-2018

#pragma once

#include "forward.h"

#include <isce/core/forward.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Orbit.h>

// isce::io
#include <isce/io/forward.h>

// isce::product
#include <isce/product/forward.h>
#include <isce/product/RadarGridParameters.h>

// isce::geometry
#include "geometry.h"

/**
 * Transformer from radar geometry coordinates to map coordinates with
 * DEM / reference altitude
 *
 * See <a href="overview_geometry.html#forwardgeom">geometry overview</a>
 * for a description of the algorithm
 */
class isce::geometry::Topo {
public:

    /**
     * Constructor using a product
     *
     * @param[in] product Input Product
     * @param[in] frequency Frequency designation
     * @param[in] nativeDoppler Flag for using native Doppler frequencies instead of zero-Doppler
     */
    Topo(const isce::product::Product &,
         char frequency = 'A',
         bool nativeDoppler = false);

    /**
     * Alternative constructor from ellipsoid orbit and radarGrid.
     *
     * @param[in] radarGrid RadarGridParameters object
     * @param[in] orbit     Orbit object
     * @param[in] ellipsoid Ellipsoid object
     * @param[in] lookSide  look direction of the radar
     * @param[in] doppler   LUT2d doppler model
     */
    Topo(const isce::product::RadarGridParameters & radarGrid,
         const isce::core::Orbit & orbit,
         const isce::core::Ellipsoid & ellipsoid,
         const isce::core::LUT2d<double> & doppler = {});

    /**
     * Constructor using core objects
     *
     * Alternative constructor from ellipsoid orbit and metadata.
     * Used for supporting VRT-formatted products.
     *
     * @param[in] ellipsoid Ellipsoid object
     * @param[in] orbit     Orbit object
     * @param[in] doppler   LUT1d doppler model
     * @param[in] meta      Metadata object with radar image parameters
     */
    Topo(const isce::core::Ellipsoid & ellipsoid,
         const isce::core::Orbit & orbit,
         const isce::core::LUT2d<double> & doppler,
         const isce::core::Metadata & meta);

    /**
     * Set convergence threshold
     *
     * @param[in] t Distance threshold to flag convergence of iterations
     */
    void threshold(double t) { _threshold = t; }

    /**
     * Set number of primary iterations
     *
     * This is the number of iterations where solution of previous solution is
     * directly used to initialize the next iteration
     *
     * @param[in] n Number of primary iterations
     */
    void numiter(int n) { _numiter = n; }

    /**
     * Set number of secondary iterations
     *
     * When we haven't converged after primary iterations, it typically means
     * that the solver is iterating between two very close solutions. In this
     * case, we use the previous 2 solutions and use the average to initialize
     * the next iteration. This is equivalent to changing the step size between
     * iterations.
     *
     * @param[in] n Number of secondary iterations
     */
    void extraiter(int n) { _extraiter = n; }

    /**
     * Set the DEM interpolation method while checking its validity
     *
     * @param[in] DEM inerpolation method
     */
    void demMethod(isce::core::dataInterpMethod method);

    /**
     * Set output coordinate system
     *
     * Set the EPSG code of the output layers and configure projection.
     * See <a href="raster.html#rasterproj">here</a> for supported projections.
     *
     * @param[in] epsgcode EPSG code of desired pixel-by-pixel outputs
     */
    void epsgOut(int epsgcode);

    /**
     * Set mask computation flag
     *
     * @param[in] mask Boolean for mask computation
     */
    void computeMask(bool mask) { _computeMask = mask; }

    /**
     * Set minimum height
     *
     * @param[in] minh Minimum altitude for the scene
     */
    void minimumHeight(double minh) { _minH = minh; }

    /**
     * Set maximum height
     *
     * @param[in] maxh Maximum altitude for the scene
     */
    void maximumHeight(double maxh) { _maxH = maxh; }

    /**
     * Set margin in decimal degrees
     *
     * @param[in] deg Margin around bounding box in decimal degrees
     */
    void decimaldegMargin(double deg) { _margin = deg; }

    // Get topo processing options

    /** Get lookSide used for processing */
    isce::core::LookSide lookSide() const { return _lookSide; }

    /** Get distance convergence threshold used for processing */
    double threshold() const { return _threshold; }

    /** Get number of primary iterations used for processing */
    int numiter() const { return _numiter; }

    /** Get number of secondary iterations used for processing*/
    int extraiter() const { return _extraiter; }

    /** Get the output coordinate system used for processing */
    int epsgOut() const { return _epsgOut; }

    /** Get the DEM interpolation method used for processing */
    isce::core::dataInterpMethod demMethod() const { return _demMethod; }

    /** Get mask computation flag */
    bool computeMask() const { return _computeMask; }

    /** Get minimum height */
    double minimumHeight() const { return _minH; }

    /** Get maximum height */
    double maximumHeight() const { return _maxH; }

    /** Get margin in decimal degrees */
    double decimaldegMargin() const { return _margin; }

    /** Get read-only reference to RadarGridParameters */
    const isce::product::RadarGridParameters & radarGridParameters() const { return _radarGrid; }

    // Get DEM bounds using first/last azimuth line and slant range bin
    void computeDEMBounds(isce::io::Raster &, DEMInterpolator &, size_t, size_t);

    /**
     * Main entry point for the module; internal creation of topo rasters
     *
     * This is the main topo driver. The pixel-by-pixel output file names are fixed for now
     * <ul>
     * <li> x.rdr - X coordinate in requested projection system (meters or degrees)
     * <li> y.rdr - Y cooordinate in requested projection system (meters or degrees)
     * <li> z.rdr - Height above ellipsoid (meters)
     * <li> inc.rdr - Incidence angle (degrees) computed from vertical at target
     * <li> hdg.rdr - Azimuth angle (degrees) computed anti-clockwise from EAST (Right hand rule)
     * <li> localInc.rdr - Local incidence angle (degrees) at target
     * <li> locaPsi.rdr - Local projection angle (degrees) at target
     * <li> simamp.rdr - Simulated amplitude image.
     * </ul>
     *
     * @param[in] demRaster input DEM raster
     * @param[in] outdir  directory to write outputs to
     */
    void topo(isce::io::Raster& demRaster, const std::string& outdir);

    /**
     * Run topo with externally created topo rasters in TopoLayers object
     *
     * @param[in] demRaster input DEM raster
     * @param[in] layers TopoLayers object for storing and writing results
     */
    void topo(isce::io::Raster & demRaster, TopoLayers & layers);

    /**
     * Run topo with externally created topo rasters; generate mask
     *
     * @param[in] demRaster input DEM raster
     * @param[in] xRaster output raster for X coordinate in requested projection system (meters or degrees)
     * @param[in] yRaster output raster for Y cooordinate in requested projection system (meters or degrees)
     * @param[in] zRaster output raster for height above ellipsoid (meters)
     * @param[in] incRaster output raster for incidence angle (degrees) computed from vertical at target
     * @param[in] hdgRaster output raster for azimuth angle (degrees) computed anti-clockwise from EAST (Right hand rule)
     * @param[in] localIncRaster output raster for local incidence angle (degrees) at target
     * @param[in] localPsiRaster output raster for local projection angle (degrees) at target
     * @param[in] simRaster output raster for simulated amplitude image.
     * @param[in] maskRaster output raster for layover/shadow mask.
     */
    void topo(isce::io::Raster& demRaster, isce::io::Raster& xRaster,
              isce::io::Raster& yRaster, isce::io::Raster& heightRaster,
              isce::io::Raster& incRaster, isce::io::Raster& hdgRaster,
              isce::io::Raster& localIncRaster,
              isce::io::Raster& localPsiRaster, isce::io::Raster& simRaster,
              isce::io::Raster& maskRaster);

    /**
     * Run topo with externally created topo rasters; generate mask
     *
     * @param[in] demRaster input DEM raster
     * @param[in] xRaster output raster for X coordinate in requested projection system (meters or degrees)
     * @param[in] yRaster output raster for Y cooordinate in requested projection system (meters or degrees)
     * @param[in] zRaster output raster for height above ellipsoid (meters)
     * @param[in] incRaster output raster for incidence angle (degrees) computed from vertical at target
     * @param[in] hdgRaster output raster for azimuth angle (degrees) computed anti-clockwise from EAST (Right hand rule)
     * @param[in] localIncRaster output raster for local incidence angle (degrees) at target
     * @param[in] localPsiRaster output raster for local projection angle (degrees) at target
     * @param[in] simRaster output raster for simulated amplitude image.
     */
    void topo(isce::io::Raster& demRaster, isce::io::Raster& xRaster,
              isce::io::Raster& yRaster, isce::io::Raster& heightRaster,
              isce::io::Raster& incRaster, isce::io::Raster& hdgRaster,
              isce::io::Raster& localIncRaster,
              isce::io::Raster& localPsiRaster, isce::io::Raster& simRaster);

    /**
     * Main entry point for the module; internal creation of topo rasters
     *
     * This is the main topo driver. The pixel-by-pixel output file names are
     * fixed for now <ul> <li> x.rdr - X coordinate in requested projection
     * system (meters or degrees) <li> y.rdr - Y cooordinate in requested
     * projection system (meters or degrees) <li> z.rdr - Height above ellipsoid
     * (meters) <li> inc.rdr - Incidence angle (degrees) computed from vertical
     * at target <li> hdg.rdr - Azimuth angle (degrees) computed anti-clockwise
     * from EAST (Right hand rule) <li> localInc.rdr - Local incidence angle
     * (degrees) at target <li> locaPsi.rdr - Local projection angle (degrees)
     * at target <li> simamp.rdr - Simulated amplitude image.
     * </ul>
     *
     * @param[in] demInterp input DEM interpolator
     * @param[in] outdir  directory to write outputs to
     */
    void topo(isce::geometry::DEMInterpolator& demInterp,
              const std::string& outdir);

    /**
     * Run topo with externally created topo rasters in TopoLayers object
     *
     * @param[in] demInterp input DEM interpolator
     * @param[in] layers TopoLayers object for storing and writing results
     */
    void topo(isce::geometry::DEMInterpolator& demInterp, TopoLayers& layers);

    /**
     * Run topo with externally created topo rasters; generate mask
     *
     * @param[in] demInterp input DEM interpolator
     * @param[in] xRaster output raster for X coordinate in requested projection
     * system (meters or degrees)
     * @param[in] yRaster output raster for Y cooordinate in requested
     * projection system (meters or degrees)
     * @param[in] zRaster output raster for height above ellipsoid (meters)
     * @param[in] incRaster output raster for incidence angle (degrees) computed
     * from vertical at target
     * @param[in] hdgRaster output raster for azimuth angle (degrees) computed
     * anti-clockwise from EAST (Right hand rule)
     * @param[in] localIncRaster output raster for local incidence angle
     * (degrees) at target
     * @param[in] localPsiRaster output raster for local projection angle
     * (degrees) at target
     * @param[in] simRaster output raster for simulated amplitude image.
     * @param[in] maskRaster output raster for layover/shadow mask.
     */
    void topo(isce::geometry::DEMInterpolator& demInterp,
              isce::io::Raster& xRaster, isce::io::Raster& yRaster,
              isce::io::Raster& heightRaster, isce::io::Raster& incRaster,
              isce::io::Raster& hdgRaster, isce::io::Raster& localIncRaster,
              isce::io::Raster& localPsiRaster, isce::io::Raster& simRaster,
              isce::io::Raster& maskRaster);

    /**
     * Run topo with externally created topo rasters; generate mask
     *
     * @param[in] demInterp input DEM interpolator
     * @param[in] xRaster output raster for X coordinate in requested projection
     * system (meters or degrees)
     * @param[in] yRaster output raster for Y cooordinate in requested
     * projection system (meters or degrees)
     * @param[in] zRaster output raster for height above ellipsoid (meters)
     * @param[in] incRaster output raster for incidence angle (degrees) computed
     * from vertical at target
     * @param[in] hdgRaster output raster for azimuth angle (degrees) computed
     * anti-clockwise from EAST (Right hand rule)
     * @param[in] localIncRaster output raster for local incidence angle
     * (degrees) at target
     * @param[in] localPsiRaster output raster for local projection angle
     * (degrees) at target
     * @param[in] simRaster output raster for simulated amplitude image.
     */
    void topo(isce::geometry::DEMInterpolator& demInterp,
              isce::io::Raster& xRaster, isce::io::Raster& yRaster,
              isce::io::Raster& heightRaster, isce::io::Raster& incRaster,
              isce::io::Raster& hdgRaster, isce::io::Raster& localIncRaster,
              isce::io::Raster& localPsiRaster, isce::io::Raster& simRaster);

    /**
     * Compute layover/shadow masks
     *
     * @param[in] layers Object containing output layers
     * @param[in] demInterp DEMInterpolator object
     * @param[in] satPosition Vector of satellite position vectors for each line
     * in block
     */
    void setLayoverShadow(TopoLayers&, DEMInterpolator&,
                          std::vector<isce::core::Vec3>&);

    // Getters for isce objects

    /** Get the orbits used for processing */
    inline const isce::core::Orbit& orbit() const { return _orbit; }
    /** Get the ellipsoid used for processing */
    inline const isce::core::Ellipsoid& ellipsoid() const { return _ellipsoid; }
    /** Get the doppler module used for processing */
    inline const isce::core::LUT2d<double>& doppler() const { return _doppler; }

private:
    /**
     * Initialize TCN basis for given azimuth line
     *
     * The module is optimized to work with range doppler coordinates. This
     * section would need to be changed to work with data in PFA coordinates
     * (not currently supported).
     *
     * @param[in] line line number of input radar geometry product
     * @param[out] pos/vel state variables needed for processing the line
     * @param[out] TCNbasis TCN basis corresponding to the state
     */
    void _initAzimuthLine(size_t line, double&,
                          isce::core::Vec3& pos, isce::core::Vec3& vel,
                          isce::core::Basis& TCNbasis);

    /**
     * Write to output layers
     *
     * Currently, local slopes are computed by simple numerical differencing.
     * In the future, we should accommodate possibility of reading in this
     * as an external layer
     *
     * @param[in] llh Lon/Lat/Hae for target
     * @param[in] layers Object containing output layers
     * @param[in] line line number to write to output
     * @param[in] pixel pixel number to write to output
     * @param[in] pos/vel state for the line under consideration
     * @param[in] TCNbasis basis for the line under consideration
     * @param[in] demInterp DEM interpolator object used to compute local slope
     */
    void _setOutputTopoLayers(isce::core::Vec3 &,
                              TopoLayers &,
                              size_t,
                              isce::core::Pixel &,
                              isce::core::Vec3& pos,
                              isce::core::Vec3& vel,
                              isce::core::Basis &,
                              DEMInterpolator &);

    /** Main entry point for the module; internal creation of topo rasters */
    template<typename T> void _topo(T& dem, const std::string& outdir);

    /** Run topo with externally created topo rasters; generate mask */
    template<typename T>
    void _topo(T& dem, isce::io::Raster& xRaster, isce::io::Raster& yRaster,
               isce::io::Raster& heightRaster, isce::io::Raster& incRaster,
               isce::io::Raster& hdgRaster, isce::io::Raster& localIncRaster,
               isce::io::Raster& localPsiRaster, isce::io::Raster& simRaster,
               isce::io::Raster& maskRaster);

    /** Run topo with externally created topo rasters; generate mask */
    template<typename T>
    void _topo(T& dem, isce::io::Raster& xRaster, isce::io::Raster& yRaster,
               isce::io::Raster& heightRaster, isce::io::Raster& incRaster,
               isce::io::Raster& hdgRaster, isce::io::Raster& localIncRaster,
               isce::io::Raster& localPsiRaster, isce::io::Raster& simRaster);

    // isce::core objects
    isce::core::Orbit _orbit;
    isce::core::Ellipsoid _ellipsoid;
    isce::core::LUT2d<double> _doppler;

    // RadarGridParameters
    isce::product::RadarGridParameters _radarGrid;

    // Optimization options
    double _threshold = 1.0e-8;   //Threshold for convergence of slant range
    int _numiter = 25;            //Number of primary iterations
    int _extraiter = 10;          //Number of secondary iterations
    double _minH = isce::core::GLOBAL_MIN_HEIGHT;   //Lowest altitude in scene (global minimum default)
    double _maxH = isce::core::GLOBAL_MAX_HEIGHT;   //Highest altitude in scene (global maximum default)
    double _margin = 0.15;        //Margin for bounding box in decimal degrees
    size_t _linesPerBlock = 1000; //Block size for processing
    bool _computeMask = true;     //Flag for generating shadow-layover mask

    isce::core::LookSide _lookSide;

    isce::core::dataInterpMethod _demMethod;

    // Output options and objects
    int _epsgOut;
    isce::core::ProjectionBase * _proj;
};

// Get inline implementations for Topo
#define ISCE_GEOMETRY_TOPO_ICC
#include "Topo.icc"
#undef ISCE_GEOMETRY_TOPO_ICC
