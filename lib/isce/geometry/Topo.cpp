// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <omp.h>
#include <valarray>
#include <algorithm>

// pyre
//#include <pyre/timers.h>

// isce::core
#include <isce/core/Constants.h>
#include <isce/core/LinAlg.h>

// isce::geometry
#include "Topo.h"

// pull in some isce::core namespaces
using isce::core::Raster;
using isce::core::Poly2d;
using isce::core::LinAlg;
using isce::core::StateVector;

// Main topo driver
void isce::geometry::Topo::
topo(Raster & demRaster,
     Poly2d & dopPoly, 
     const std::string outdir) {
    
    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.geometry.Topo");
    pyre::journal::info_t info("isce.geometry.Topo");

    // First check that variables have been initialized
    checkInitialization(info);
    
    // Output layers
    TopoLayers layers(_meta.width);
    
    // Create rasters for individual layers
    Raster latRaster = Raster(outdir + "/lat.rdr", _meta.width, _meta.length, 1,
        GDT_Float64, "ENVI");
    Raster lonRaster = Raster(outdir + "/lon.rdr", _meta.width, _meta.length, 1,
        GDT_Float64, "ENVI");
    Raster heightRaster = Raster(outdir + "/z.rdr", _meta.width, _meta.length, 1,
        GDT_Float64, "ENVI");
    Raster incRaster = Raster(outdir + "/inc.rdr", _meta.width, _meta.length, 1,
        GDT_Float32, "ENVI");
    Raster hdgRaster = Raster(outdir + "/hdg.rdr", _meta.width, _meta.length, 1,
        GDT_Float32, "ENVI");
    Raster localIncRaster = Raster(outdir + "/localInc.rdr", _meta.width, _meta.length, 1,
        GDT_Float32, "ENVI");
    Raster localPsiRaster = Raster(outdir + "/localPsi.rdr", _meta.width, _meta.length, 1,
        GDT_Float32, "ENVI");
    Raster simRaster = Raster(outdir + "/simamp.rdr", _meta.width, _meta.length, 1,
        GDT_Float32, "ENVI");
    
    // Create and start a timer
    //pyre::timer_t topoTimer("isce.geometry.Topo");
    //topoTimer.start();

    // Create a DEM interpolator
    DEMInterpolator demInterp(-500.0);
    // Load DEM subset for SLC image bounds
    _computeDEMBounds(demRaster, demInterp, dopPoly);

    // Compute max and mean DEM height for the subset
    float demmax, dem_avg;
    demInterp.computeHeightStats(demmax, dem_avg, info);

    // For each line
    size_t totalconv = 0;
    for (int line = 0; line < _meta.length; line++) {

        // Periodic diagnostic printing
        if ((line % 1000) == 0) {
            info 
                << "Processing line: " << line << " " << pyre::journal::newline
                << "Dopplers near mid far: "
                << dopPoly.eval(0, 0) << " "
                << dopPoly.eval(0, (_meta.width / 2) - 1) << " "
                << dopPoly.eval(0, _meta.width - 1) << " "
                << pyre::journal::newline;
        }

        // Initialize orbital data for this azimuth line
        Basis basis;
        StateVector state;
        _initAzimuthLine(line, state, basis);

        // Compute velocity magnitude
        const double satVmag = LinAlg::norm(state.velocity());
        
        // For each slant range bin
        //#pragma omp parallel reduction(+:totalconv)
        const double radians = M_PI / 180.0;
        for (int rbin = 0; rbin < _meta.width; ++rbin) {

            // Get current slant range
            const double rng = _meta.rangeFirstSample + rbin*_meta.slantRangePixelSpacing;

            // Get current Doppler value
            const double dopfact = (0.5 * _meta.radarWavelength 
                                 * (dopPoly.eval(0, rbin) / satVmag)) * rng;

            // Store slant range bin data in Pixel
            Pixel pixel(rng, dopfact, rbin);

            // Initialize LLH to middle of input DEM and average height
            cartesian_t llh = {radians*demInterp.midLat(), radians*demInterp.midLon(), dem_avg};

            // Perform rdr->geo iterations
            int geostat = Geometry::rdr2geo(
                pixel, basis, state, _ellipsoid, _ptm, demInterp, llh, _meta.lookSide,
                _threshold, _numiter, _extraiter
            );
            totalconv += geostat;

            // Save data in output arrays
            _setOutputTopoLayers(llh, layers, pixel, state, basis, demInterp);

        } //end OMP for loop
        
        // Write out line of data for every product
        latRaster.setLine(layers.lat(), line);
        lonRaster.setLine(layers.lon(), line);
        heightRaster.setLine(layers.z(), line);
        incRaster.setLine(layers.inc(), line);
        hdgRaster.setLine(layers.hdg(), line);
        localIncRaster.setLine(layers.localInc(), line);
        localPsiRaster.setLine(layers.localPsi(), line);
        simRaster.setLine(layers.sim(), line);
    }

    //// Print out timing information and reset
    //topoTimer.stop();
    //info << "Elapsed processing time: " << topoTimer.read() << " sec"
    //     << pyre::journal::newline;
    //topoTimer.reset();

    // Print out convergence statistics
    info << "Total convergence: " << totalconv << " out of "
         << (_meta.width * _meta.length) << pyre::journal::endl;

}

// Perform data initialization for a given azimuth line
void isce::geometry::Topo::
_initAzimuthLine(int line, StateVector & state, Basis & basis) {

    // Get satellite azimuth time
    const double tline = _meta.sensingStart.secondsSinceEpoch() 
                      + (_meta.numberAzimuthLooks * (line/_meta.prf));
    
    // Get state vector
    cartesian_t xyzsat, velsat;
    int stat = _orbit.interpolate(tline, xyzsat, velsat, _orbitMethod);
    if (stat != 0) {
        pyre::journal::error_t error("isce.core.Topo._initAzimuthLine");
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

    // Get TCN basis using satellite basis
    cartesian_t that, chat, nhat;
    _ellipsoid.TCNbasis(xyzsat, velsat, that, chat, nhat);
    // Save to basis
    basis.that(that);
    basis.chat(chat);
    basis.nhat(nhat);

    // Convert satellite position to lat-lon
    cartesian_t llhsat;
    _ellipsoid.xyzToLatLon(xyzsat, llhsat);

    // Set peg point right below satellite
    _peg.lat = llhsat[0];
    _peg.lon = llhsat[1];
    _peg.hdg = _meta.pegHeading;
    _ptm.radarToXYZ(_ellipsoid, _peg);
}

// Get DEM bounds using first/last azimuth line and slant range bin
void isce::geometry::Topo::
_computeDEMBounds(Raster & demRaster, DEMInterpolator & demInterp, Poly2d & dopPoly) {

    // Initialize journal
    pyre::journal::warning_t warning("isce.core.Topo");

    cartesian_t llh, satLLH;
    StateVector state;
    Basis basis;

    // Initialize geographic bounds
    double min_lat = 10000.0;
    double max_lat = -10000.0;
    double min_lon = 10000.0;
    double max_lon = -10000.0;
    const double degrees = 180.0 / M_PI;

    // Loop over first and last azimuth line
    for (int line = 0; line < 2; line++) {

        // Initialize orbit data for this azimuth line
        int lineIndex = line * _meta.numberAzimuthLooks * (_meta.length - 1);
        _initAzimuthLine(lineIndex, state, basis);

        // Compute satellite velocity and height
        const double satVmag = LinAlg::norm(state.velocity());
        _ellipsoid.xyzToLatLon(state.position(), satLLH);

        // Loop over starting and ending slant range
        for (int ind = 0; ind < 2; ++ind) {

            // Get proper slant range bin and Doppler factor
            int rbin = ind * (_meta.width - 1);
            double rng = _meta.rangeFirstSample + rbin*_meta.slantRangePixelSpacing;
            double dopfact = (0.5 * _meta.radarWavelength * (dopPoly.eval(0, rbin) 
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
                    Geometry::rdr2geo(pixel, basis, state, _ellipsoid, _ptm, constDEM,
                                      llh, _meta.lookSide, _threshold, 1, 0);
                }
                // Update bounds
                min_lat = std::min(min_lat, llh[0]*degrees);
                max_lat = std::max(max_lat, llh[0]*degrees);
                min_lon = std::min(min_lon, llh[1]*degrees);
                max_lon = std::max(max_lon, llh[1]*degrees);
            }
        }
    }

    // Account for margins
    min_lon -= MARGIN;
    max_lon += MARGIN;
    min_lat -= MARGIN;
    max_lat += MARGIN;

    // Extract DEM subset
    demInterp.loadDEM(demRaster, min_lon, max_lon, min_lat, max_lat, _demMethod);
    demInterp.declare();
}

// Generate output topo layers
void isce::geometry::Topo::
_setOutputTopoLayers(cartesian_t & targetLLH, TopoLayers & layers, Pixel & pixel,
                     StateVector & state, Basis & basis, DEMInterpolator & demInterp) {

    cartesian_t targetXYZ, targetSCH, satToGround, satLLH, enu, vhat;
    cartmat_t enumat, xyz2enu;
    const double degrees = 180.0 / M_PI;
    const double radians = M_PI / 180.0;

    // Unpack the lat/lon values and convert to degrees
    const double lat = degrees * targetLLH[0];
    const double lon = degrees * targetLLH[1];
    // Unpack the range pixel data
    const size_t bin = pixel.bin();
    const double rng = pixel.range();
    const double dopfact = pixel.dopfact();

    // Set outputs for LLH
    layers.lat(bin, lat);
    layers.lon(bin, lon);
    layers.z(bin, targetLLH[2]);

    // Convert llh->xyz for ground point
    _ellipsoid.latLonToXyz(targetLLH, targetXYZ);
    // Convert xyz->SCH for ground point
    _ptm.convertSCHtoXYZ(targetSCH, targetXYZ, isce::core::XYZ_2_SCH);
    const double zsch = targetSCH[2];

    // Compute vector from satellite to ground point
    LinAlg::linComb(1.0, targetXYZ, -1.0, state.position(), satToGround);

    // Compute unit velocity vector
    LinAlg::unitVec(state.velocity(), vhat);
    
    // Computation in ENU coordinates around target
    LinAlg::enuBasis(targetLLH[0], targetLLH[1], enumat);
    LinAlg::tranMat(enumat, xyz2enu);
    LinAlg::matVec(xyz2enu, satToGround, enu);
    const double cosalpha = std::abs(enu[2]) / LinAlg::norm(enu);

    // Compute satellite height above ellipsoid
    _ellipsoid.xyzToLatLon(state.position(), satLLH);
    const double satHeight = satLLH[2];

    // Look angles
    double aa = satHeight + _ptm.radcur;
    double bb = _ptm.radcur + zsch;
    double costheta = 0.5 * ((aa / rng) + (rng / aa) - ((bb / aa) * (bb / rng)));
    double sintheta = std::sqrt(1.0 - costheta * costheta);
    double gamma = rng * costheta;
    double alpha = dopfact - gamma*LinAlg::dot(basis.nhat(), vhat)
                 / LinAlg::dot(vhat, basis.that());
    double beta = -1 * _meta.lookSide * std::sqrt(
                   rng * rng * sintheta * sintheta - alpha * alpha);
    
    // LOS vectors
    layers.inc(bin, std::acos(cosalpha) * degrees);
    layers.hdg(bin, (std::atan2(-enu[1], -enu[0]) - (0.5*M_PI)) * degrees);

    // East-west slope using central difference
    aa = demInterp.interpolate(lat, lon - demInterp.deltaLon());
    bb = demInterp.interpolate(lat, lon + demInterp.deltaLon());
    gamma = lat * radians;
    alpha = ((bb - aa) * degrees) 
          / (2.0 * _ellipsoid.rEast(gamma) * demInterp.deltaLon());
    
    // North-south slope using central difference
    aa = demInterp.interpolate(lat - demInterp.deltaLat(), lon);
    bb = demInterp.interpolate(lat + demInterp.deltaLat(), lon);
    beta = ((bb - aa) * degrees) 
         / (2.0 * _ellipsoid.rNorth(gamma) * demInterp.deltaLat());

    // Compute local incidence angle
    const double enunorm = LinAlg::norm(enu);
    for (int idx = 0; idx < 3; ++idx) {
        enu[idx] = enu[idx] / enunorm;
    }
    costheta = ((enu[0] * alpha) + (enu[1] * beta) - enu[2])
             / std::sqrt(1.0 + (alpha * alpha) + (beta * beta));
    layers.localInc(bin, std::acos(costheta)*degrees);

    // Compute amplitude simulation
    sintheta = std::sqrt(1.0 - (costheta * costheta));
    bb = sintheta + 0.1 * costheta;
    layers.sim(bin, std::log10(std::abs(0.01 * costheta / (bb * bb * bb))));

    // Calculate psi angle between image plane and local slope
    cartesian_t n_img, n_imghat, n_img_enu;
    LinAlg::cross(satToGround, state.velocity(), n_img);
    LinAlg::unitVec(n_img, n_imghat);
    LinAlg::scale(n_imghat, -1*_meta.lookSide);
    LinAlg::matVec(xyz2enu, n_imghat, n_img_enu);
    cartesian_t n_trg_enu = {-alpha, -beta, 1.0};
    const double cospsi = LinAlg::dot(n_trg_enu, n_img_enu)
          / (LinAlg::norm(n_trg_enu) * LinAlg::norm(n_img_enu));
    layers.localPsi(bin, std::acos(cospsi) * degrees);
}

// end of file
