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
#include <vector>
#include <valarray>
#include <algorithm>

// isce::core
#include <isce/core/Constants.h>
#include <isce/core/LinAlg.h>
#include <isce/core/Utilities.h>

// isce::geometry
#include "Topo.h"

// pull in some isce::core namespaces
using isce::core::Basis;
using isce::core::Pixel;
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

    // Initialize a TopoLayers object to handle block data and raster data
    TopoLayers layers;

    // Create rasters for individual layers (provide output raster sizes)
    layers.initRasters(outdir, _radarGridParameters.width(), _radarGridParameters.length(),
                       _computeMask);

    // Call topo with layers
    topo(demRaster, layers);

    } // end Topo scope to release raster resources

    // Write out multi-band topo VRT
    std::vector<Raster> rasterTopoVec = {
        Raster(outdir + "/x.rdr" ),
        Raster(outdir + "/y.rdr" ),
        Raster(outdir + "/z.rdr" ),
        Raster(outdir + "/inc.rdr" ),
        Raster(outdir + "/hdg.rdr" ),
        Raster(outdir + "/localInc.rdr" ),
        Raster(outdir + "/localPsi.rdr" ),
        Raster(outdir + "/simamp.rdr" )
    };

    // Add optional mask raster
    if (_computeMask) {
        rasterTopoVec.push_back(Raster(outdir + "/mask.rdr" ));
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
  * @param[in] simRaster output raster for simulated amplitude image. 
  * @param[in] maskRaster output raster for layover/shadow mask. */
void isce::geometry::Topo::
topo(Raster & demRaster, Raster & xRaster, Raster & yRaster, Raster & heightRaster,
     Raster & incRaster, Raster & hdgRaster, Raster & localIncRaster, Raster & localPsiRaster,
     Raster & simRaster, Raster & maskRaster) {

    // Initialize a TopoLayers object to handle block data and raster data
    TopoLayers layers;

    // Create rasters for individual layers (provide output raster sizes)
    layers.setRasters(xRaster, yRaster, heightRaster, incRaster, hdgRaster, localIncRaster,
                      localPsiRaster, simRaster, maskRaster);
    // Indicate a mask raster has been provided for writing
    computeMask(true);

    // Call topo with layers
    topo(demRaster, layers);
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

    // Initialize a TopoLayers object to handle block data and raster data
    TopoLayers layers;

    // Create rasters for individual layers (provide output raster sizes)
    layers.setRasters(xRaster, yRaster, heightRaster, incRaster, hdgRaster, localIncRaster,
                      localPsiRaster, simRaster);
    // Indicate no mask raster has been provided for writing
    computeMask(false);

    // Call topo with layers
    topo(demRaster, layers);
}

/** @param[in] demRaster input DEM raster
  * @param[in] layers TopoLayers object for storing and writing results
  */
void isce::geometry::Topo::
topo(Raster & demRaster, TopoLayers & layers) {

    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.geometry.Topo");
    pyre::journal::info_t info("isce.geometry.Topo");

    // Create and start a timer
    auto timerStart = std::chrono::steady_clock::now();

    // Create a DEM interpolator
    DEMInterpolator demInterp(-500.0, _demMethod);

    // Compute number of blocks needed to process image
    size_t nBlocks = _radarGridParameters.length() / _linesPerBlock;
    if ((_radarGridParameters.length() % _linesPerBlock) != 0)
        nBlocks += 1;

    // Cache range bounds for diagnostics
    const double startingRange = _radarGridParameters.startingRange();
    const double endingRange = _radarGridParameters.endingRange();
    const double midRange = _radarGridParameters.midRange();

    // Loop over blocks
    size_t totalconv = 0;
    for (size_t block = 0; block < nBlocks; ++block) {

        // Get block extents
        size_t lineStart, blockLength;
        lineStart = block * _linesPerBlock;
        if (block == (nBlocks - 1)) {
            blockLength = _radarGridParameters.length() - lineStart;
        } else {
            blockLength = _linesPerBlock;
        }

        // Diagnostics
        const double tblock = _radarGridParameters.sensingTime(lineStart);
        info << "Processing block: " << block << " " << pyre::journal::newline
             << "  - line start: " << lineStart << pyre::journal::newline
             << "  - line end  : " << lineStart + blockLength << pyre::journal::newline
             << "  - dopplers near mid far: "
             << _doppler.eval(tblock, startingRange) << " "
             << _doppler.eval(tblock, midRange) << " " 
             << _doppler.eval(tblock, endingRange) << " "
             << pyre::journal::endl;

        // Load DEM subset for SLC image block
        computeDEMBounds(demRaster, demInterp, lineStart, blockLength);

        // Compute max and mean DEM height for the subset
        float demmax, dem_avg;
        demInterp.computeHeightStats(demmax, dem_avg, info);
        // Reset reference height for DEMInterpolator
        demInterp.refHeight(dem_avg);

        // Set output block sizes in layers
        layers.setBlockSize(blockLength, _radarGridParameters.width());

        // Allocate vector for storing satellite position for each line
        std::vector<cartesian_t> satPosition(blockLength);

        // For each line in block
        double tline;
        for (size_t blockLine = 0; blockLine < blockLength; ++blockLine) {

            // Global line index
            size_t line = lineStart + blockLine;

            // Initialize orbital data for this azimuth line
            Basis TCNbasis;
            StateVector state;
            _initAzimuthLine(line, tline, state, TCNbasis);
            satPosition[blockLine] = state.position();

            // Compute velocity magnitude
            const double satVmag = LinAlg::norm(state.velocity());

            // For each slant range bin
            #pragma omp parallel for reduction(+:totalconv)
            for (size_t rbin = 0; rbin < _radarGridParameters.width(); ++rbin) {

                // Get current slant range
                const double rng = _radarGridParameters.slantRange(rbin);

                // Get current Doppler value
                const double dopfact = (0.5 * _radarGridParameters.wavelength()
                                     * (_doppler.eval(tline, rng) / satVmag)) * rng;

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

        // Compute layover/shadow masks for the block
        if (_computeMask) {
            setLayoverShadow(layers, demInterp, satPosition);
        }

        // Write out block of data for all topo layers
        layers.writeData(0, lineStart);    
        
    } // end for loop blocks

    // Print out convergence statistics
    info << "Total convergence: " << totalconv << " out of "
         << _radarGridParameters.size() << pyre::journal::endl;

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
_initAzimuthLine(size_t line, double & tline, StateVector & state, Basis & TCNbasis) {

    // Get satellite azimuth time
    tline = _radarGridParameters.sensingTime(line);

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
    const int rskip = _radarGridParameters.width() / 10;

    // Construct vectors of range/azimuth indices traversing the perimeter of the radar frame

    // Top edge
    std::vector<int> azInd, rgInd;
    for (int j = 0; j < _radarGridParameters.width(); j += rskip) {
        azInd.push_back(0);
        rgInd.push_back(j);
    }

    // Right edge
    for (int i = 0; i < blockLength; i += askip) {
        azInd.push_back(i);
        rgInd.push_back(_radarGridParameters.width());
    }

    // Bottom edge
    for (int j = _radarGridParameters.width(); j > 0; j -= rskip) {
        azInd.push_back(blockLength - 1);
        rgInd.push_back(j);
    }

    // Left edge
    for (int i = blockLength; i > 0; i -= askip) {
        azInd.push_back(i);
        rgInd.push_back(0);
    }

    // Loop over the indices
    double tline;
    for (size_t i = 0; i < rgInd.size(); ++i) {

        // Convert az index to absolute line index
        size_t lineIndex = lineOffset + azInd[i] * _radarGridParameters.numberAzimuthLooks();

         // Initialize orbit data for this azimuth line
        StateVector state;
        Basis TCNbasis;
        _initAzimuthLine(lineIndex, tline, state, TCNbasis);

        // Compute satellite velocity and height
        cartesian_t satLLH;
        const double satVmag = LinAlg::norm(state.velocity());
        _ellipsoid.xyzToLonLat(state.position(), satLLH);

        // Get proper slant range and Doppler factor
        const size_t rbin = rgInd[i];
        double rng = _radarGridParameters.slantRange(rbin);
        double dopfact = (0.5 * _radarGridParameters.wavelength() * (_doppler.eval(tline, rng)
                        / satVmag)) * rng;
        // Store in Pixel object
        Pixel pixel(rng, dopfact, rbin);

        // Run topo for one iteration for two different heights
        cartesian_t llh = {0, 0, 0};
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

    // Compute cross-track range
    layers.crossTrack(line, bin, -1.0 * _lookSide * LinAlg::dot(satToGround, TCNbasis.x1()));

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

/** @param[in] layers Object containing output layers
 *  @param[in] demInterp DEMInterpolator object
 *  @param[in] satPosition Vector of cartesian_t of satellite position for each line in block
 *
 *  Compute layer and shadow mask following the logic from ISCE 2
 */
void isce::geometry::Topo::
setLayoverShadow(TopoLayers & layers, DEMInterpolator & demInterp,
                 std::vector<cartesian_t> & satPosition) {

    // Cache the width of the block
    const int width = layers.width();
    // Compute layover on oversampled grid
    const int gridWidth = 2 * width;

    // Allocate working valarrays
    std::valarray<double> x(width), y(width), ctrack(width), ctrackGrid(gridWidth);
    std::valarray<double> slantRange(width), slantRangeGrid(gridWidth);
    std::valarray<short> maskGrid(gridWidth);

    // Pre-compute slantRange grid used for all lines
    for (int i = 0; i < width; ++i) {
        slantRange[i] = _radarGridParameters.slantRange(i);
    }
   
    // Initialize mask to zero for this block 
    layers.mask() = 0;

    // Loop over lines in block
    #pragma omp parallel for firstprivate(x, y, ctrack, ctrackGrid, \
                                          slantRangeGrid, maskGrid)
    for (size_t line = 0; line < layers.length(); ++line) {

        // Cache satellite position for this line
        const cartesian_t xyzsat = satPosition[line];

        // Copy cross-track, x, and y values for the line
        for (int i = 0; i < width; ++i) {
            ctrack[i] = layers.crossTrack(line, i);
            x[i] = layers.x(line, i);
            y[i] = layers.y(line, i);
        }
 
        // Sort ctrack, x, and y by values in ctrack
        isce::core::insertionSort(ctrack, x, y);
         
        // Create regular grid for cross-track values
        const double cmin = ctrack.min();// - demInterp.maxHeight();
        const double cmax = ctrack.max();// + demInterp.maxHeight();
        isce::core::linspace<double>(cmin, cmax, ctrackGrid);

        // Interpolate DEM to regular cross-track grid
        for (int i = 0; i < gridWidth; ++i) {

            // Compute nearest ctrack index for current ctrackGrid value
            const double crossTrack = ctrackGrid[i];
            int k = isce::core::binarySearch(ctrack, crossTrack);
            // Adjust edges if necessary
            if (k == (width - 1)) {
                k = width - 2;
            } else if (k < 0) {
                k = 0;
            }

            // Bilinear interpolation to estimate DEM x/y coordinates
            const double c1 = ctrack[k];
            const double c2 = ctrack[k+1];
            const double frac1 = (c2 - crossTrack) / (c2 - c1);
            const double frac2 = (crossTrack - c1) / (c2 - c1);
            const double x_grid = x[k] * frac1 + x[k+1] * frac2;
            const double y_grid = y[k] * frac1 + y[k+1] * frac2;

            // Interpolate DEM at x/y
            const float z_grid = demInterp.interpolateXY(x_grid, y_grid);

            // Convert DEM XYZ to ECEF XYZ
            cartesian_t llh, xyz, satToGround;
            cartesian_t demXYZ{x_grid, y_grid, z_grid};
            _proj->inverse(demXYZ, llh);
            _ellipsoid.lonLatToXyz(llh, xyz);

            // Compute and save slant range
            LinAlg::linComb(1.0, xyz, -1.0, xyzsat, satToGround);
            slantRangeGrid[i] = LinAlg::norm(satToGround);
        }
        
        // Now sort cross-track grid in terms of slant range grid
        isce::core::insertionSort(slantRangeGrid, ctrackGrid);

        // Traverse from near range to far range on original spacing for shadow detection
        double minIncAngle = layers.inc(line, 0);
        for (int i = 1; i < width; ++i) {
            const double inc = layers.inc(line, i);
            // Test shadow
            if (inc <= minIncAngle) {
                layers.mask(line, i, isce::core::SHADOW_VALUE);
            } else {
                minIncAngle = inc;
            }
        }
    
        // Traverse from far range to near range on original spacing for shadow detection
        double maxIncAngle = layers.inc(line, width - 1);
        for (int i = width - 2; i >= 0; --i) {
            const double inc = layers.inc(line, i);
            // Test shadow
            if (inc >= maxIncAngle) {
                layers.mask(line, i, isce::core::SHADOW_VALUE);
            } else {
                maxIncAngle = inc;
            }
        }

        // Traverse from near range to far range on grid spacing for layover detection
        maskGrid = 0;
        double minCrossTrack = ctrackGrid[0];
        for (int i = 1; i < gridWidth; ++i) {
            const double crossTrack = ctrackGrid[i];
            // Test layover
            if (crossTrack <= minCrossTrack) {
                maskGrid[i] = isce::core::LAYOVER_VALUE;
            } else {
                minCrossTrack = crossTrack;
            }
        }

        // Traverse from far range to near range on grid spacing for layover detection
        double maxCrossTrack = ctrackGrid[gridWidth - 1];
        for (int i = gridWidth - 2; i >= 0; --i) {
            const double crossTrack = ctrackGrid[i];
            // Test layover
            if (crossTrack >= maxCrossTrack) {
                maskGrid[i] = isce::core::LAYOVER_VALUE;
            } else {
                maxCrossTrack = crossTrack;
            }
        }

        // Resample maskGrid to original spacing
        for (int i = 0; i < gridWidth; ++i) {
            if (maskGrid[i] > 0) {
                // Find index in original grid spacing
                int k = isce::core::binarySearch(slantRange, slantRangeGrid[i]);
                if (k < 0 || k >= width) continue;
                // Update it
                const short maskval = layers.mask(line, k);
                if (maskval < isce::core::LAYOVER_VALUE) {
                    layers.mask(line, k, maskval + isce::core::LAYOVER_VALUE);
                }
            }
        }
    } // end loop lines
}

// end of file
