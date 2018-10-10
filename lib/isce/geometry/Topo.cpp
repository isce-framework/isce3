// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <future>
#include <valarray>
#include <algorithm>

// isce::core
#include <isce/core/Constants.h>
#include <isce/core/LinAlg.h>

// isce::geometry
#include "Topo.h"

// pull in some isce::core namespaces
using isce::core::Basis;
using isce::core::Pixel;
using isce::core::Poly2d;
using isce::core::LinAlg;
using isce::core::StateVector;
using isce::io::Raster;

// Main topo driver; internally create topo rasters
/** @param[in] demRaster input DEM raster
  * @param[in] outdir  directory to write outputs to
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
  * </ul>*/
void isce::geometry::Topo::
topo(Raster & demRaster,
     const std::string outdir) {

    { // Topo scope for creating output rasters

    // Create rasters for individual layers
    Raster xRaster = Raster(outdir + "/x.rdr", _mode.width(), _mode.length(), 1,
        GDT_Float64, "ISCE");
    Raster yRaster = Raster(outdir + "/y.rdr", _mode.width(), _mode.length(), 1,
        GDT_Float64, "ISCE");
    Raster heightRaster = Raster(outdir + "/z.rdr", _mode.width(), _mode.length(), 1,
        GDT_Float64, "ISCE");
    Raster incRaster = Raster(outdir + "/inc.rdr", _mode.width(), _mode.length(), 1,
        GDT_Float32, "ISCE");
    Raster hdgRaster = Raster(outdir + "/hdg.rdr", _mode.width(), _mode.length(), 1,
        GDT_Float32, "ISCE");
    Raster localIncRaster = Raster(outdir + "/localInc.rdr", _mode.width(), _mode.length(), 1,
        GDT_Float32, "ISCE");
    Raster localPsiRaster = Raster(outdir + "/localPsi.rdr", _mode.width(), _mode.length(), 1,
        GDT_Float32, "ISCE");
    Raster simRaster = Raster(outdir + "/simamp.rdr", _mode.width(), _mode.length(), 1,
        GDT_Float32, "ISCE");

    // Call topo with rasters
    topo(demRaster, xRaster, yRaster, heightRaster, incRaster, hdgRaster, localIncRaster,
         localPsiRaster, simRaster);

    } // end Topo scope to release raster resources

    // Write out multi-band topo VRT
    const std::vector<Raster> rasterTopoVec = {
        Raster(outdir + "/x.rdr" ),
        Raster(outdir + "/y.rdr" ),
        Raster(outdir + "/z.rdr" ),
        Raster(outdir + "/inc.rdr" ),
        Raster(outdir + "/hdg.rdr" ),
        Raster(outdir + "/localInc.rdr" ),
        Raster(outdir + "/localPsi.rdr" ),
        Raster(outdir + "/simamp.rdr" )
    };
    Raster vrt = Raster(outdir + "/topo.vrt", rasterTopoVec );
    // Set its EPSG code
    vrt.setEPSG(_epsgOut);
}

/** @param[in] demRaster input DEM raster
  * @param[in] xRaster output raster for X coordinate in requested projection system 
                   (meters or degrees)
  * @param[in] yRaster output raster for Y cooordinate in requested projection system
                   (meters or degrees)
  * @param[in] zRaster output raster for height above ellipsoid (meters)
  * @param[in] incRaster output raster for incidence angle (degrees) computed from vertical 
               at target
  * @param[in] hdgRaster output raster for azimuth angle (degrees) computed anti-clockwise 
               from EAST (Right hand rule)
  * @param[in] localIncRaster output raster for local incidence angle (degrees) at target
  * @param[in] localPsiRaster output raster for local projection angle (degrees) at target
  * @param[in] simRaster output raster for simulated amplitude image. */
void isce::geometry::Topo::
topo(Raster & demRaster, Raster & xRaster, Raster & yRaster, Raster & heightRaster,
     Raster & incRaster, Raster & hdgRaster, Raster & localIncRaster, Raster & localPsiRaster,
     Raster & simRaster) {

    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.geometry.Topo");
    pyre::journal::info_t info("isce.geometry.Topo");

    // First check that variables have been initialized
    checkInitialization(info); 

    // Create and start a timer
    auto timerStart = std::chrono::steady_clock::now();

    // Create a DEM interpolator
    DEMInterpolator demInterp(-500.0, _demMethod);

    // Compute number of blocks needed to process image
    size_t nBlocks = _mode.length() / _linesPerBlock;
    if ((_mode.length() % _linesPerBlock) != 0)
        nBlocks += 1;

    // Loop over blocks
    size_t totalconv = 0;
    for (size_t block = 0; block < nBlocks; ++block) {

        // Get block extents
        size_t lineStart, blockLength;
        lineStart = block * _linesPerBlock;
        if (block == (nBlocks - 1)) {
            blockLength = _mode.length() - lineStart;
        } else {
            blockLength = _linesPerBlock;
        }

        // Diagnostics
        info << "Processing block: " << block << " " << pyre::journal::newline
             << "  - line start: " << lineStart << pyre::journal::newline
             << "  - line end  : " << lineStart + blockLength << pyre::journal::newline
             << "  - dopplers near mid far: "
             << _doppler.eval(0, 0) << " "
             << _doppler.eval(0, (_mode.width() / 2) - 1) << " "
             << _doppler.eval(0, _mode.width() - 1) << " "
             << pyre::journal::endl;

        // Load DEM subset for SLC image block
        computeDEMBounds(demRaster, demInterp, lineStart, blockLength);

        // Compute max and mean DEM height for the subset
        float demmax, dem_avg;
        demInterp.computeHeightStats(demmax, dem_avg, info);
        // Reset reference height for DEMInterpolator
        demInterp.refHeight(dem_avg);

        // Output layers for block
        TopoLayers layers(blockLength, _mode.width());

        // For each line in block
        for (size_t blockLine = 0; blockLine < blockLength; ++blockLine) {

            // Global line index
            size_t line = lineStart + blockLine;

            // Initialize orbital data for this azimuth line
            Basis TCNbasis;
            StateVector state;
            _initAzimuthLine(line, state, TCNbasis);

            // Compute velocity magnitude
            const double satVmag = LinAlg::norm(state.velocity());

            // For each slant range bin
            #pragma omp parallel for reduction(+:totalconv)
            for (size_t rbin = 0; rbin < _mode.width(); ++rbin) {

                // Get current slant range
                const double rng = _mode.startingRange() + rbin * _mode.rangePixelSpacing();

                // Get current Doppler value
                const double dopfact = (0.5 * _mode.wavelength()
                                     * (_doppler.eval(0, rbin) / satVmag)) * rng;

                // Store slant range bin data in Pixel
                Pixel pixel(rng, dopfact, rbin);

                // Initialize LLH to middle of input DEM and average height
                cartesian_t llh = demInterp.midLonLat();

                // Perform rdr->geo iterations
                int geostat = rdr2geo(
                    pixel, TCNbasis, state, _ellipsoid, demInterp, llh, _lookSide,
                    _threshold, _numiter, _extraiter
                );
                totalconv += geostat;

                // Save data in output arrays
                _setOutputTopoLayers(llh, layers, blockLine, pixel, state, TCNbasis, demInterp);

            } // end OMP for loop pixels in block
        } // end for loop lines in block

        // Write out block of data for every product
        xRaster.setBlock(layers.x(), 0, lineStart, _mode.width(), blockLength);
        yRaster.setBlock(layers.y(), 0, lineStart, _mode.width(), blockLength);
        heightRaster.setBlock(layers.z(), 0, lineStart, _mode.width(), blockLength);
        incRaster.setBlock(layers.inc(), 0, lineStart, _mode.width(), blockLength);
        hdgRaster.setBlock(layers.hdg(), 0, lineStart, _mode.width(), blockLength);
        localIncRaster.setBlock(layers.localInc(), 0, lineStart, _mode.width(), blockLength);
        localPsiRaster.setBlock(layers.localPsi(), 0, lineStart, _mode.width(), blockLength);
        simRaster.setBlock(layers.sim(), 0, lineStart, _mode.width(), blockLength);

    } // end for loop blocks

    // Print out convergence statistics
    info << "Total convergence: " << totalconv << " out of "
         << (_mode.width() * _mode.length()) << pyre::journal::endl;

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
        timerEnd - timerStart).count();
    info << "Elapsed processing time: " << elapsed << " sec"
         << pyre::journal::newline;
}

/** @param[in] line line number of input radar geometry product
 * @param[out] state store state variables needed for processing the line
 * @param[out] TCNbasis TCN basis corresponding to the state
 *
 * The module is optimized to work with range doppler coordinates. This section would need to be changed to work with data in PFA coordinates (not currently supported). */
void isce::geometry::Topo::
_initAzimuthLine(size_t line, StateVector & state, Basis & TCNbasis) {

    // Get satellite azimuth time
    const double tline = _mode.startAzTime().secondsSinceEpoch(_refEpoch)
                      + (_mode.numberAzimuthLooks() * (line / _mode.prf()));

    // Get state vector
    cartesian_t xyzsat, velsat;
    int stat = _orbit.interpolate(tline, xyzsat, velsat, _orbitMethod);
    if (stat != 0) {
        pyre::journal::error_t error("isce.geometry.Topo._initAzimuthLine");
        error
            << pyre::journal::at(__HERE__)
            << "Error in Topo::topo - Error getting state vector for bounds computation."
            << pyre::journal::newline
            << " - requested time: " << tline << pyre::journal::newline
            << " - bounds: " << _orbit.UTCtime[0] << " -> " << _orbit.UTCtime[_orbit.nVectors-1]
            << pyre::journal::endl;
    }
    // Save state vector
    state.position(xyzsat);
    state.velocity(velsat);

    // Get geocentric TCN basis using satellite basis
    geocentricTCN(state, TCNbasis);
    
}

// Get DEM bounds using first/last azimuth line and slant range bin
void isce::geometry::Topo::
computeDEMBounds(Raster & demRaster, DEMInterpolator & demInterp, size_t lineOffset,
                 size_t blockLength) {

    // Initialize journal
    pyre::journal::warning_t warning("isce.core.Topo");

    // Initialize geographic bounds
    double min_lat = 10000.0;
    double max_lat = -10000.0;
    double min_lon = 10000.0;
    double max_lon = -10000.0;

    // Skip factors along azimuth and range
    const int askip = std::max((int) blockLength / 10, 1);
    const int rskip = _mode.width() / 10;

    // Construct vectors of range/azimuth indices traversing the perimeter of the radar frame

    // Top edge
    std::vector<int> azInd, rgInd;
    for (int j = 0; j < _mode.width(); j += rskip) {
        azInd.push_back(0);
        rgInd.push_back(j);
    }

    // Right edge
    for (int i = 0; i < blockLength; i += askip) {
        azInd.push_back(i);
        rgInd.push_back(_mode.width());
    }

    // Bottom edge
    for (int j = _mode.width(); j > 0; j -= rskip) {
        azInd.push_back(blockLength - 1);
        rgInd.push_back(j);
    }

    // Left edge
    for (int i = blockLength; i > 0; i -= askip) {
        azInd.push_back(i);
        rgInd.push_back(0);
    }

    // Loop over the indices
    for (size_t i = 0; i < rgInd.size(); ++i) {

        // Convert az index to absolute line index
        size_t lineIndex = lineOffset + azInd[i] * _mode.numberAzimuthLooks();

         // Initialize orbit data for this azimuth line
        StateVector state;
        Basis TCNbasis;
        _initAzimuthLine(lineIndex, state, TCNbasis);

        // Compute satellite velocity and height
        cartesian_t satLLH;
        const double satVmag = LinAlg::norm(state.velocity());
        _ellipsoid.xyzToLonLat(state.position(), satLLH);

        // Get proper slant range and Doppler factor
        const size_t rbin = rgInd[i];
        double rng = _mode.startingRange() + rbin * _mode.rangePixelSpacing();
        double dopfact = (0.5 * _mode.wavelength() * (_doppler.eval(0, rbin)
                        / satVmag)) * rng;
        // Store in Pixel object
        Pixel pixel(rng, dopfact, rbin);

        // Run topo for one iteration for two different heights
        cartesian_t llh;
        std::array<double, 2> testHeights = {MIN_H, MAX_H};
        for (int k = 0; k < 2; ++k) {

            // If slant range vector doesn't hit ground, pick nadir point
            if (rng <= (satLLH[2] - testHeights[k] + 1.0)) {
                for (int idx = 0; idx < 3; ++idx) {
                    llh[idx] = satLLH[idx];
                }
                warning << "Possible near nadir imaging" << pyre::journal::endl;
            } else {
                // Create dummy DEM interpolator with constant height
                DEMInterpolator constDEM(testHeights[k]);
                // Run radar->geo for 1 iteration
                rdr2geo(pixel, TCNbasis, state, _ellipsoid, constDEM, llh,
                        _lookSide, _threshold, 1, 0);
            }

            // Update bounds
            min_lat = std::min(min_lat, llh[1]);
            max_lat = std::max(max_lat, llh[1]);
            min_lon = std::min(min_lon, llh[0]);
            max_lon = std::max(max_lon, llh[0]);
        }
    }

    // Account for margins
    min_lon -= MARGIN;
    max_lon += MARGIN;
    min_lat -= MARGIN;
    max_lat += MARGIN;

    // Extract DEM subset
    demInterp.loadDEM(demRaster, min_lon, max_lon, min_lat, max_lat,
                      demRaster.getEPSG());
    demInterp.declare();
}

/** @param[in] llh Lon/Lat/Hae for target 
 * @param[in] layers Object containing output layers
 * @param[in] line line number to write to output
 * @param[in] pixel pixel number to write to output
 * @param[in] state state for the line under consideration
 * @param[in] TCNbasis basis for the line under consideration
 * @param[in] demInterp DEM interpolator object used to compute local slope
 *
 * Currently, local slopes are computed by simple numerical differencing. In the future, we should accommodate possibility of reading in this as an external layer*/
void isce::geometry::Topo::
_setOutputTopoLayers(cartesian_t & targetLLH, TopoLayers & layers, size_t line,
                     Pixel & pixel, StateVector & state, Basis & TCNbasis,
                     DEMInterpolator & demInterp) {

    cartesian_t targetXYZ, satToGround, enu, vhat;
    cartmat_t enumat, xyz2enu;
    const double degrees = 180.0 / M_PI;

    // Unpack the range pixel data
    const size_t bin = pixel.bin();

    // Convert lat/lon values to output coordinate system
    cartesian_t xyzOut;
    _proj->forward(targetLLH, xyzOut);
    const double x = xyzOut[0];
    const double y = xyzOut[1];

    // Set outputs
    layers.x(line, bin, x);
    layers.y(line, bin, y);
    layers.z(line, bin, targetLLH[2]);

    // Convert llh->xyz for ground point
    _ellipsoid.lonLatToXyz(targetLLH, targetXYZ);

    // Compute vector from satellite to ground point
    LinAlg::linComb(1.0, targetXYZ, -1.0, state.position(), satToGround);

    // Compute unit velocity vector
    LinAlg::unitVec(state.velocity(), vhat);

    // Computation in ENU coordinates around target
    LinAlg::enuBasis(targetLLH[1], targetLLH[0], enumat);
    LinAlg::tranMat(enumat, xyz2enu);
    LinAlg::matVec(xyz2enu, satToGround, enu);
    const double cosalpha = std::abs(enu[2]) / LinAlg::norm(enu);
    
    // LOS vectors
    layers.inc(line, bin, std::acos(cosalpha) * degrees);
    layers.hdg(line, bin, (std::atan2(-enu[1], -enu[0]) - (0.5*M_PI)) * degrees);

    // East-west slope using central difference
    double aa = demInterp.interpolateXY(x - demInterp.deltaX(), y);
    double bb = demInterp.interpolateXY(x + demInterp.deltaX(), y);
    double gamma = targetLLH[1];
    double alpha = ((bb - aa) * degrees) / (2.0 * _ellipsoid.rEast(gamma) * demInterp.deltaX());

    // North-south slope using central difference
    aa = demInterp.interpolateXY(x, y - demInterp.deltaY());
    bb = demInterp.interpolateXY(x, y + demInterp.deltaY());
    double beta = ((bb - aa) * degrees) / (2.0 * _ellipsoid.rNorth(gamma) * demInterp.deltaY());

    // Compute local incidence angle
    const double enunorm = LinAlg::norm(enu);
    for (int idx = 0; idx < 3; ++idx) {
        enu[idx] = enu[idx] / enunorm;
    }
    double costheta = ((enu[0] * alpha) + (enu[1] * beta) - enu[2])
                     / std::sqrt(1.0 + (alpha * alpha) + (beta * beta));
    layers.localInc(line, bin, std::acos(costheta)*degrees);

    // Compute amplitude simulation
    double sintheta = std::sqrt(1.0 - (costheta * costheta));
    bb = sintheta + 0.1 * costheta;
    layers.sim(line, bin, std::log10(std::abs(0.01 * costheta / (bb * bb * bb))));

    // Calculate psi angle between image plane and local slope
    cartesian_t n_img, n_imghat, n_img_enu;
    LinAlg::cross(satToGround, state.velocity(), n_img);
    LinAlg::unitVec(n_img, n_imghat);
    LinAlg::scale(n_imghat, -1*_lookSide);
    LinAlg::matVec(xyz2enu, n_imghat, n_img_enu);
    cartesian_t n_trg_enu = {-alpha, -beta, 1.0};
    const double cospsi = LinAlg::dot(n_trg_enu, n_img_enu)
          / (LinAlg::norm(n_trg_enu) * LinAlg::norm(n_img_enu));
    layers.localPsi(line, bin, std::acos(cospsi) * degrees);
}

// end of file
