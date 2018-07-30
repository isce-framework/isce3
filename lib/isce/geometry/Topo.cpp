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

// Main topo driver
void isce::geometry::Topo::
topo(Raster & demRaster,
     const std::string outdir) {

    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.geometry.Topo");
    pyre::journal::info_t info("isce.geometry.Topo");

    // First check that variables have been initialized
    checkInitialization(info);

    // Output layers
    TopoLayers layers(_mode.width());

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

    // Create and start a timer
    auto timerStart = std::chrono::steady_clock::now();

    // Create a DEM interpolator
    DEMInterpolator demInterp(-500.0);
    // Load DEM subset for SLC image bounds
    _computeDEMBounds(demRaster, demInterp);

    // Compute max and mean DEM height for the subset
    float demmax, dem_avg;
    std::cout << "computing height stats\n";
    demInterp.computeHeightStats(demmax, dem_avg, info);

    // For each line
    size_t totalconv = 0;
    for (size_t line = 0; line < _mode.length(); line++) {

        // Periodic diagnostic printing
        if ((line % 1000) == 0) {
            info
                << "Processing line: " << line << " " << pyre::journal::newline
                << "Dopplers near mid far: "
                << _doppler.eval(0, 0) << " "
                << _doppler.eval(0, (_mode.width() / 2) - 1) << " "
                << _doppler.eval(0, _mode.width() - 1) << " "
                << pyre::journal::endl;
        }

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
            cartesian_t llh = demInterp.midLonLat(dem_avg);

            // Perform rdr->geo iterations
            int geostat = rdr2geo(
                pixel, TCNbasis, state, _ellipsoid, demInterp, llh, _lookSide,
                _threshold, _numiter, _extraiter
            );
            totalconv += geostat;

            // Save data in output arrays
            _setOutputTopoLayers(llh, layers, pixel, state, TCNbasis, demInterp);

        } //end OMP for loop

        // Write out line of data for every product
        xRaster.setLine(layers.x(), line);
        yRaster.setLine(layers.y(), line);
        heightRaster.setLine(layers.z(), line);
        incRaster.setLine(layers.inc(), line);
        hdgRaster.setLine(layers.hdg(), line);
        localIncRaster.setLine(layers.localInc(), line);
        localPsiRaster.setLine(layers.localPsi(), line);
        simRaster.setLine(layers.sim(), line);
    }

    // Print out convergence statistics
    info << "Total convergence: " << totalconv << " out of "
         << (_mode.width() * _mode.length()) << pyre::journal::endl;

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
        timerEnd - timerStart).count();
    info << "Elapsed processing time: " << elapsed << " sec"
         << pyre::journal::newline;

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

// Perform data initialization for a given azimuth line
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
_computeDEMBounds(Raster & demRaster, DEMInterpolator & demInterp) {

    // Initialize journal
    pyre::journal::warning_t warning("isce.core.Topo");

    cartesian_t llh, satLLH;
    StateVector state;
    Basis TCNbasis;

    // Initialize geographic bounds
    double min_lat = 10000.0;
    double max_lat = -10000.0;
    double min_lon = 10000.0;
    double max_lon = -10000.0;

    // Loop over first and last azimuth line
    for (int line = 0; line < 2; line++) {

        // Initialize orbit data for this azimuth line
        int lineIndex = line * _mode.numberAzimuthLooks() * (_mode.length() - 1);
        _initAzimuthLine(lineIndex, state, TCNbasis);

        // Compute satellite velocity and height
        const double satVmag = LinAlg::norm(state.velocity());
        _ellipsoid.xyzToLonLat(state.position(), satLLH);

        // Loop over starting and ending slant range
        for (int ind = 0; ind < 2; ++ind) {

            // Get proper slant range bin and Doppler factor
            int rbin = ind * (_mode.width() - 1);
            double rng = _mode.startingRange() + rbin * _mode.rangePixelSpacing();
            double dopfact = (0.5 * _mode.wavelength() * (_doppler.eval(0, rbin)
                            / satVmag)) * rng;
            // Store in Pixel object
            Pixel pixel(rng, dopfact, rbin);

            // Run topo for one iteration for two different heights
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
    }

    // Account for margins
    min_lon -= MARGIN;
    max_lon += MARGIN;
    min_lat -= MARGIN;
    max_lat += MARGIN;

    // Extract DEM subset
    demInterp.loadDEM(demRaster, min_lon, max_lon, min_lat, max_lat, _demMethod,
                      demRaster.getEPSG());
    demInterp.declare();
}

// Generate output topo layers
void isce::geometry::Topo::
_setOutputTopoLayers(cartesian_t & targetLLH, TopoLayers & layers, Pixel & pixel,
                     StateVector & state, Basis & TCNbasis, DEMInterpolator & demInterp) {

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
    layers.x(bin, x);
    layers.y(bin, y);
    layers.z(bin, targetLLH[2]);

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
    layers.inc(bin, std::acos(cosalpha) * degrees);
    layers.hdg(bin, (std::atan2(-enu[1], -enu[0]) - (0.5*M_PI)) * degrees);

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
    layers.localInc(bin, std::acos(costheta)*degrees);

    // Compute amplitude simulation
    double sintheta = std::sqrt(1.0 - (costheta * costheta));
    bb = sintheta + 0.1 * costheta;
    layers.sim(bin, std::log10(std::abs(0.01 * costheta / (bb * bb * bb))));

    // Calculate psi angle between image plane and local slope
    cartesian_t n_img, n_imghat, n_img_enu;
    LinAlg::cross(satToGround, state.velocity(), n_img);
    LinAlg::unitVec(n_img, n_imghat);
    LinAlg::scale(n_imghat, -1*_lookSide);
    LinAlg::matVec(xyz2enu, n_imghat, n_img_enu);
    cartesian_t n_trg_enu = {-alpha, -beta, 1.0};
    const double cospsi = LinAlg::dot(n_trg_enu, n_img_enu)
          / (LinAlg::norm(n_trg_enu) * LinAlg::norm(n_img_enu));
    layers.localPsi(bin, std::acos(cospsi) * degrees);
}

// end of file
