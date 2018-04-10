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

// isce::core
#include "Constants.h"
#include "Geometry.h"
#include "Peg.h"
#include "Pegtrans.h"
#include "LinAlg.h"
#include "Topo.h"

// Main topo driver
void isce::core::Topo::
topo(Raster & demRaster,
     Poly2d & dopPoly, 
     Poly2d & slrngPoly,
     const std::string outdir) {

    isceLib::mat_t enumat, xyz2enu;
    isceLib::cartesian_t xyz, llh, delta, llh_prev, xyz_prev;
    isceLib::cartesian_t xyzsat, velsat, llhsat, enu, that, nhat, chat;
    isceLib::cartesian_t vhat, n_img, n_imghat, n_img_enu, n_trg_enu;
    
    // Local geometry-type objects
    Peg peg;
    Pegtrans ptm;
    const double degrees = 180.0 / M_PI;
    const double radians = M_PI / 180.0;

    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.core.Topo");
    pyre::journal::info_t info("isce.core.Topo");

    // Set up DEM interpolation method
    if ((dem_method != SINC_METHOD) && (dem_method != BILINEAR_METHOD) &&
        (dem_method != BICUBIC_METHOD) && (dem_method != NEAREST_METHOD) &&
        (dem_method != AKIMA_METHOD) && (dem_method != BIQUINTIC_METHOD)) {
        printf("Error in Topo::topo - Undefined interpolation method.\n");
        exit(1);
    }
    
    // Set up orbit interpolation method
    _checkOrbitMethod();

    owidth = (2 * _meta.width) + 1;
    totalconv = 0;
    
    // Working vectors to hold line of results
    std::valarray<double> lat(_meta.width), lon(_meta.width), z(_meta.width), zsch(_meta.width),
        converge(_meta.width);
    std::valarray<float> distance(_meta.width), incang(_meta.width), hdgang(_meta.width),
        sarSim(_meta.width), localIncAng(_meta.width), localPsiAng(_meta.width);

    // Initialize (so no scoping errors), resize only if needed
    std::valarray<double> omask(0), orng(0), ctrack(0), oview(0), mask(0); 
    
    // Output raster objects
    Raster latRaster = Raster(outdir + "/lat.rdr", _meta.width, _meta.length, 1,
        GDT_Float64, "ENVI");
    Raster lonRaster = Raster(outdir + "/lon.rdr", _meta.width, _meta.length, 1,
        GDT_Float64, "ENVI");
    Raster heightRaster = Raster(outdir + "/z.rdr", _meta.width, _meta.length, 1,
        GDT_Float64, "ENVI");
    Raster losRaster = Raster(outdir + "/z.rdr", _meta.width, _meta.length, 2,
        GDT_Float32, "ENVI");
    Raster incRaster = Raster(outdir + "/inc.rdr", _meta.width, _meta.length, 2,
        GDT_Float32, "ENVI");
    Raster simRaster = Raster(outdir + "/simamp.rdr", _meta.width, _meta.length, 1,
        GDT_Float32, "ENVI");

    // Create a DEM interpolator
    DEMInterpolator demInterp(demRaster);
    // Load DEM subset for SLC image bounds
    _computeDEMBounds(demRaster, demInterp, slrngPoly, dopPoly);

    // Compute max and mean DEM height for the subset
    info << "Computing DEM statistics" << pyre::journal::newline << pyre::journal::newline;
    float demmax, dem_avg;
    demInterp.computeHeightStats(demmax, dem_avg);
    
    info << "Max DEM height: " << demmax << pyre::journal::newline
         << "Primary iterations: " << numiter << pyre::journal::newline
         << "Secondary iterations: " << extraiter << pyre::journal::newline
         << "Distance threshold: " << thresh << pyre::journal::newline << pyre::journal::newline
         << "Average DEM height: " << dem_avg << pyre::journal::endl;;

    // For each line
    double satHeight = 0.0;
    for (int line = 0; line < _meta.length; line++) {

        // Periodic diagnostic printing
        if ((line % 1000) == 0) {
            info 
                << "Processing line: " << line << " " << satVmag << pyre::journal::newline
                << "Dopplers near mid far: "
                << dopPoly.eval(0, 0) << " "
                << dopPoly.eval(0, (_meta.width / 2) - 1) << " "
                << dopPoly.eval(0, _meta.width - 1) << " "
                << pyre::journal::endl;
        }

        // Initialize orbital data for this azimuth line
        _initAzimuthLine(line, peg, ptm, satVmag, satHeight, that, nhat, chat);
        
        // For each slant range bin
        #pragma omp parallel for private( \
            firstprivate(delta) \
            reduction(+:totalconv)
        for (int pixel = 0; pixel < _meta.width; ++pixel) {

            // Cartesian vectors and matrices
            cartesian_t llh, xyz, sch, satToGround, enu, n_img, n_imghat, n_img_enu;
            cartmat_t enumat, xyz2enu;

            // Get current slant range
            double rng = slrngPoly.eval(0, pixel);

            // Get current Doppler value
            double dopfact = (0.5 * _meta.radarWavelength 
                           * (dopPoly.eval(0, pixel) / satVmag)) * rng;

            // Initialize LLH to middle of input DEM and average height
            llh[0] = demInterp.midLat();
            llh[1] = demInterp.midLon();
            llh[2] = dem_avg;

            // Perform iterations
            Geometry::rdr2geo(rng, dopfact, that, chat, nhat, _ellipsoid, ptm, demInterp,
                              llh, delta, zsch_pixel, numiter, extraiter);

            // Set result
            lat[pixel] = llh[0] * degrees;
            lon[pixel] = llh[1] * degrees;
            z[pixel] = llh[2];

            // ----------------------------------------------------------------------
            // The rest of the loop is dedicated to generating output layers
            // ----------------------------------------------------------------------

            // Convert llh->xyz
            _ellipsoid.latLonToXyz(llh, xyz);
            // Convert xyz->SCH
            ptm.convertSCHtoXYZ(xyz, sch, XYZ_2_SCH);
            zsch[pixel] = sch[2];
            // Compute vector from satellite to ground point
            LinAlg::linComb(1.0, xyz, -1.0, satPos, satToGround);
            
            // Computation in ENU coordinates around target
            LinAlg::enuBasis(llh[0], llh[1], enumat);
            LinAlg::tranMat(enumat, xyz2enu);
            LinAlg::matVec(xyz2enu, satToGround, enu);
            const double cosalpha = std::abs(enu[2]) / LinAlg::norm(enu);

            // Look angles
            const double aa = satHeight + ptm.radcur;
            const double bb = ptm.radcur + zsch[pixel];
            const double costheta = 0.5 * ((aa / rng) + (rng / aa) - ((bb / aa) * (bb / rng)));
            const double sintheta = sqrt(1.0 - (costheta * costheta));
            const double gamma = rng * costheta;
            const double alpha = dopfact - gamma*LinAlg::dot(nhat, satVel)
                               / LinAlg::dot(satVel, that);
            const double beta = -1 * _meta.lookSide * std::sqrt(
                                rng * rng * sintheta * sintheta - alpha * alpha);
            
            // LOS vectors
            incang[pixel] = std::acos(cosalpha) * degrees;
            hdgang[pixel] = (std::atan2(-enu[1], -enu[0]) - (0.5*M_PI)) * degrees;

            // East-west slope using central difference
            aa = demInterp.interpolate(lat[pixel], lon[pixel] - demInterp.deltaLon());
            bb = demInterp.interpolate(lat[pixel], lon[pixel] + demInterp.deltaLon());
            const double gamm = lat[pixel] * radians;
            const double alpha = ((bb - aa) * degrees) 
                               / (2.0 * _ellipsoid.rEast(gamm) * demInterp.deltaLon());
            
            // North-south slope using central difference
            aa = demInterp.interpolate(lat[pixel] - demInterp.deltaLat(), lon[pixel]);
            bb = demInterp.interpolate(lat[pixel] + demInterp.deltaLat(), lon[pixel]);
            const double beta = ((bb - aa) * degrees) 
                              / (2.0 * _ellipsoid.rNorth(gamm) * demInterp.daltaLat());

            // Compute local incidence angle
            const double enunorm = LinAlg::norm(enu);
            for (int idx = 0; idx < 3; ++idx) {
                enu[idx] = enu[idx] / enunorm;
            }
            const double incAngle = std::acos(((enu[0] * alpha) + (enu[1] * beta) - enu[2])
                                  / std::sqrt(1.0 + (alpha * alpha) + (beta * beta)));
            localIncAng[pixel] = incAngle * degrees;

            // Compute amplitude simulation
            const double sinInc = std::sin(incAngle);
            bb = sintheta + 0.1 * costheta;
            sarSim[pixel] = std::log10(std::abs(0.01 * std::cos(incAngle) / (bb * bb * bb)));

            // Calculate psi angle between image plane and local slope
            LinAlg::cross(satToGround, satVel, n_img);
            LinAlg::unitVec(n_img, n_imghat);
            LinAlg::scale(n_imghat, -1*_meta.lookSide);
            LinAlg::matVec(xyz2enu, n_imghat, n_img_enu);
            cartesian_t n_trg_enu = {-alpha, -beta, 1.0};
            const double cospsi = LinAlg::dot(n_trg_enu, n_img_enu)
                  / (LinAlg::norm(n_trg_enu) * LinAlg::norm(n_img_enu));
            localPsiAng[pixel] = std::acos(cospsi) * degrees;

        } //end OMP for loop
        
        // Write out line of data
        latRaster.setLine(line, lat);
        lonRaster.setLine(line, lon);
        heightRaster.setLine(line, z);
        incRaster.setLine(line, incang);
        hdgRaster.setLine(line, hdgang);
        localIncRaster.setLine(line, localIncAng);
        localPsiRaster.setLine(line, localPsiAng);
        simRaster.setLine(line, sarSim);

    }

    //printf("Total convergence: %d out of %d.\n", totalconv, (_meta.width * _meta.length));
}

// Perform data initialization for a given azimuth line
void isce::core::Topo::
_initAzimuthLine(int line, Peg & peg, Pegtrans & ptm, double & satVmag, double & satHeight,
    cartesian_t & that, cartesian_t & nhat, cartesian_t & chat) {

    cartesian_t xyzsat, velsat, vhat, llhsat;

    // Get satellite azimuth time
    tline = _meta.sensingStart.secondsOfDay() + (_meta.numberAzimuthLooks * (line / _meta.prf));
    
    // Get state vector
    int stat = _orbit.interpolate(tline, xyzsat, velsat, _orbit_method);
    if (stat != 0) {
        pyre::journal::error_t error("isce.core.Topo._initAzimuthLine");
        error
            << pyre::journal::at(__HERE__)
            << "Error in Topo::topo - Error getting state vector for bounds computation."
            << pyre::journal::newline
            << " - requested time: " tline << pyre::journal::newline
            << " - bounds: " << orb.UTCtime[0] << " -> " << orb.UTCtime[orb.nVectors-1]
            << pyre::journal::endl;
    }
    // Unit velocity vector
    LinAlg::unitVec(velsat, vhat);
    // Velocity magnitude
    satVmag = LinAlg::norm(velsat)

    // Convert satellite position to lat-lon
    _ellipsoid.xyzToLatLon(xyzsat, llhsat);
    satHeight = llhsat[2];

    // Get TCN basis using satellite basis
    _ellipsoid.TCNbasis(xyzsat, velsat, that, chat, nhat);

    // Set peg point right below satellite
    peg.lat = llhsat[0];
    peg.lon = llhsat[1];
    peg.hdg = _meta.pegHeading;
    ptm.radarToXYZ(_ellipsoid, peg);
}

// Get DEM bounds using first/last azimuth line and slant range bin
void isce::core::Topo::
_computeDEMBounds(Raster & demRaster, Poly2d & slrngPoly, Poly2d & dopPoly,
                  int & ustarty, int & ustartx, int & udemwidth, int & udemlength) {

    // Initialize journal
    pyre::journal::warning_t warning("isce.core.Topo");

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
        _initAzimuthLine(lineIndex, peg, ptm, satVmag, satHeight, that, nhat, chat);

        // Loop over starting and ending slant range
        for (int ind = 0; ind < 2; ++ind) {

            // Get proper slant range bin and Doppler factor
            int pixel = ind * (_meta.width - 1);
            double rng = slrngPoly.eval(0, pixel);
            double dopfact = (0.5 * _meta.radarWavelength * (dopPoly.eval(0, pixel) 
                            / satVmag)) * rng;

            // Run topo for one iteration for two different heights
            std::array<double, 2> testHeights = {MIN_H, MAX_H};
            for (int k = 0; k < 2; ++k) {
                // If slant range vector doesn't hit ground, pick nadir point
                if (rng <= (llhsat[2] - testHeights[k] + 1.0)) {
                    for (int idx = 0; idx < 3; ++idx) {
                        llh[idx] = llhsat[idx];
                    }
                    warning << "Possible near nadir imaging" << pyre::journal::endl;
                } else {
                    // Create dummy DEM interpolator with constant height
                    DEMInterpolator constDEM(testHeights[k]);
                    // Run rdr2geo
                    Geometry::rdr2geo(rng, dopfact, that, chat, nhat, _ellipsoid, ptm, llh, 1);
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

    // Subset DEM
    demInterp.loadDEM(min_lon, max_lon, min_lat, max_lat);
    demInterp.declare();
}

// Check orbit method for validity
void isce::core::Topo::
_checkOrbitMethod() {
    if (_orbit_method == HERMITE_METHOD) {
        if (orb.nVectors < 4) {
            pyre::journal::error_t error("isce.core.Topo");
            error
                << pyre::journal::at(__HERE__)
                << "Error in Topo::topo - Need at least 4 state vectors for using "
                << "hermite polynomial interpolation."
                << pyre::journal::endl;
        }
    } else if (_orbit_method == SCH_METHOD) {
        if (orb.nVectors < 4) {
            pyre::journal::error_t error("isce.core.Topo");
            error
                << pyre::journal::at(__HERE__)
                << "Error in Topo::topo - Need at least 4 state vectors for using "
                << "SCH interpolation."
                << pyre::journal::endl;
        }
    } else if (_orbit_method == LEGENDRE_METHOD) {
        if (orb.nVectors < 9) {
            pyre::journal::error_t error("isce.core.Topo");
            error
                << pyre::journal::at(__HERE__)
                << "Error in Topo::topo - Need at least 9 state vectors for using "
                << "legendre polynomial interpolation."
                << pyre::journal::endl;
        }
    } else {
        pyre::journal::error_t error("isce.core.Topo");
        error
            << pyre::journal::at(__HERE__)
            << "Error in Topo::topo - Undefined orbit interpolation method."
            << pyre::journal::endl;
    }
}

// Generate output topo layers
void isce::core::Topo::
_generateOutputLayers() {

    // Convert llh->xyz
    _ellipsoid.latLonToXyz(llh, xyz);
    // Convert xyz->SCH
    ptm.convertSCHtoXYZ(xyz, sch, XYZ_2_SCH);
    zsch[pixel] = sch[2];
    // Compute vector from satellite to ground point
    LinAlg::linComb(1.0, xyz, -1.0, satPos, satToGround);
    
    // Computation in ENU coordinates around target
    LinAlg::enuBasis(llh[0], llh[1], enumat);
    LinAlg::tranMat(enumat, xyz2enu);
    LinAlg::matVec(xyz2enu, satToGround, enu);
    const double cosalpha = std::abs(enu[2]) / LinAlg::norm(enu);

    // Look angles
    const double aa = satHeight + ptm.radcur;
    const double bb = ptm.radcur + zsch[pixel];
    const double costheta = 0.5 * ((aa / rng) + (rng / aa) - ((bb / aa) * (bb / rng)));
    const double sintheta = sqrt(1.0 - (costheta * costheta));
    const double gamma = rng * costheta;
    const double alpha = dopfact - gamma*LinAlg::dot(nhat, satVel)
                       / LinAlg::dot(satVel, that);
    const double beta = -1 * _meta.lookSide * std::sqrt(
                        rng * rng * sintheta * sintheta - alpha * alpha);
    
    // LOS vectors
    incang[pixel] = std::acos(cosalpha) * degrees;
    hdgang[pixel] = (std::atan2(-enu[1], -enu[0]) - (0.5*M_PI)) * degrees;

    // East-west slope using central difference
    aa = demInterp.interpolate(lat[pixel], lon[pixel] - demInterp.deltaLon());
    bb = demInterp.interpolate(lat[pixel], lon[pixel] + demInterp.deltaLon());
    const double gamm = lat[pixel] * radians;
    const double alpha = ((bb - aa) * degrees) 
                       / (2.0 * _ellipsoid.rEast(gamm) * demInterp.deltaLon());
    
    // North-south slope using central difference
    aa = demInterp.interpolate(lat[pixel] - demInterp.deltaLat(), lon[pixel]);
    bb = demInterp.interpolate(lat[pixel] + demInterp.deltaLat(), lon[pixel]);
    const double beta = ((bb - aa) * degrees) 
                      / (2.0 * _ellipsoid.rNorth(gamm) * demInterp.daltaLat());

    // Compute local incidence angle
    const double enunorm = LinAlg::norm(enu);
    for (int idx = 0; idx < 3; ++idx) {
        enu[idx] = enu[idx] / enunorm;
    }
    const double incAngle = std::acos(((enu[0] * alpha) + (enu[1] * beta) - enu[2])
                          / std::sqrt(1.0 + (alpha * alpha) + (beta * beta)));
    localIncAng[pixel] = incAngle * degrees;

    // Compute amplitude simulation
    const double sinInc = std::sin(incAngle);
    bb = sintheta + 0.1 * costheta;
    sarSim[pixel] = std::log10(std::abs(0.01 * std::cos(incAngle) / (bb * bb * bb)));

    // Calculate psi angle between image plane and local slope
    LinAlg::cross(satToGround, satVel, n_img);
    LinAlg::unitVec(n_img, n_imghat);
    LinAlg::scale(n_imghat, -1*_meta.lookSide);
    LinAlg::matVec(xyz2enu, n_imghat, n_img_enu);
    cartesian_t n_trg_enu = {-alpha, -beta, 1.0};
    const double cospsi = LinAlg::dot(n_trg_enu, n_img_enu)
          / (LinAlg::norm(n_trg_enu) * LinAlg::norm(n_img_enu));
    localPsiAng[pixel] = std::acos(cospsi) * degrees;

}

// end of file
