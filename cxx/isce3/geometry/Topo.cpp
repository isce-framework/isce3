#include "Topo.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <valarray>
#include <vector>

// isce3::core
#include <isce3/core/Basis.h>
#include <isce3/core/Constants.h>
#include <isce3/core/Pixel.h>
#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Utilities.h>

#include <isce3/product/RadarGridProduct.h>

// isce3::geometry
#include <isce3/geometry/loadDem.h>
#include "DEMInterpolator.h"
#include "TopoLayers.h"

// pull in some isce3::core namespaces
using isce3::core::Basis;
using isce3::core::Mat3;
using isce3::core::Pixel;
using isce3::core::Vec3;
using isce3::io::Raster;

isce3::geometry::Topo::
Topo(const isce3::product::RadarGridProduct & product,
     char frequency,
     bool nativeDoppler)
:
    _radarGrid(product, frequency)
{
    // Copy orbit and doppler
    _orbit = product.metadata().orbit();
    if (nativeDoppler) {
        _doppler = product.metadata().procInfo().dopplerCentroid(frequency);
    }

    // Make an ellipsoid manually
    _ellipsoid = isce3::core::Ellipsoid(isce3::core::EarthSemiMajorAxis,
                                       isce3::core::EarthEccentricitySquared);

    // Adjust block length based in input SLC length
    _linesPerBlock = std::min(_radarGrid.length(), _linesPerBlock);
}

// Main topo driver; internally create topo rasters
template<typename T>
void isce3::geometry::Topo::_topo(T& dem, const std::string& outdir) {
    { // Topo scope for creating output rasters
        // Initialize a TopoLayers object to handle block data and raster data
        // Create rasters for individual layers (provide output raster sizes)
        TopoLayers layers(outdir, _radarGrid.length(), _radarGrid.width(),
                          _linesPerBlock, _computeMask);

        // Call topo with layers
        topo(dem, layers);
    } // end Topo scope to release raster resources

    // Write out multi-band topo VRT
    std::vector<Raster> rasterTopoVec = {
            Raster(outdir + "/x.rdr"),        Raster(outdir + "/y.rdr"),
            Raster(outdir + "/z.rdr"),        Raster(outdir + "/inc.rdr"),
            Raster(outdir + "/hdg.rdr"),      Raster(outdir + "/localInc.rdr"),
            Raster(outdir + "/localPsi.rdr"), Raster(outdir + "/simamp.rdr")};

    // Add optional mask raster
    if (_computeMask) {
        rasterTopoVec.push_back(Raster(outdir + "/layoverShadowMask.rdr"));
    };

    Raster vrt = Raster(outdir + "/topo.vrt", rasterTopoVec);
    // Set its EPSG code
    vrt.setEPSG(_epsgOut);
}

// Run topo with externally created topo rasters
template<typename T>
void isce3::geometry::Topo::_topo(T& dem, Raster* xRaster, Raster* yRaster,
                                 Raster* heightRaster, Raster* incRaster,
                                 Raster* hdgRaster, Raster* localIncRaster,
                                 Raster* localPsiRaster, Raster* simRaster,
                                 Raster* maskRaster) {
    // Initialize a TopoLayers object to handle block data and raster data
    // Create rasters for individual layers (provide output raster sizes)
    TopoLayers layers(_linesPerBlock, xRaster, yRaster, heightRaster, incRaster,
                      hdgRaster, localIncRaster, localPsiRaster, simRaster,
                      maskRaster);

    // Set computeMask flag by pointer value
    computeMask(maskRaster != nullptr);

    // Call topo with layers
    topo(dem, layers);
}

void isce3::geometry::Topo::
topo(Raster & demRaster, TopoLayers & layers)
{
    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.geometry.Topo");
    pyre::journal::info_t info("isce.geometry.Topo");

    // Create and start a timer
    auto timerStart = std::chrono::steady_clock::now();

    // Create a DEM interpolator
    DEMInterpolator demInterp(-500.0, _demMethod);

    // Compute number of blocks needed to process image
    size_t nBlocks = _radarGrid.length() / _linesPerBlock;
    if ((_radarGrid.length() % _linesPerBlock) != 0)
        nBlocks += 1;

    // Cache range bounds for diagnostics
    const double startingRange = _radarGrid.startingRange();
    const double endingRange = _radarGrid.endingRange();
    const double midRange = _radarGrid.midRange();

    info << "DEM EPSG: " << demRaster.getEPSG() << pyre::journal::newline;
    info << "Output EPSG: " << _epsgOut << pyre::journal::endl;

    // Loop over blocks
    size_t totalconv = 0;
    for (size_t block = 0; block < nBlocks; ++block) {

        // Get block extents
        size_t lineStart, blockLength;
        lineStart = block * _linesPerBlock;
        if (block == (nBlocks - 1)) {
            blockLength = _radarGrid.length() - lineStart;
        } else {
            blockLength = std::min(_linesPerBlock,  _radarGrid.length());
        }

        // Diagnostics
        const double tblock = _radarGrid.sensingTime(lineStart);
        info << "Processing block: " << block + 1 << " " << pyre::journal::newline
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
        float demmin, demmax, dem_avg;
        demInterp.computeMinMaxMeanHeight(demmin, demmax, dem_avg);
        // Reset reference height for DEMInterpolator
        demInterp.refHeight(dem_avg);

        // Reset output block sizes in layers
        layers.setBlockSize(blockLength, _radarGrid.width());

        // Allocate vector for storing satellite position for each line
        std::vector<Vec3> satPosition(blockLength);

        // For each line in block
        double tline;
        for (size_t blockLine = 0; blockLine < blockLength; ++blockLine) {

            if (blockLine % std::max((int) (blockLength / 100), 1) == 0)
                printf("\rTopo progress (block %d/%d): %d%%",
                       (int) block + 1, (int) nBlocks,
                       (int) (blockLine * 1e2 / blockLength)),
                       fflush(stdout);

            // Global line index
            size_t line = lineStart + blockLine;

            // Initialize orbital data for this azimuth line
            Basis TCNbasis;
            Vec3 pos, vel;
            _initAzimuthLine(line, tline, pos, vel, TCNbasis);

            satPosition[blockLine] = pos;

            // Compute velocity magnitude
            const double satVmag = vel.norm();

            // For each slant range bin
            #pragma omp parallel for reduction(+:totalconv)
            for (size_t rbin = 0; rbin < _radarGrid.width(); ++rbin) {

                // Get current slant range
                const double rng = _radarGrid.slantRange(rbin);

                // Get current Doppler value
                const double dopfact = (0.5 * _radarGrid.wavelength()
                                     * (_doppler.eval(tline, rng) / satVmag)) * rng;

                // Store slant range bin data in Pixel
                Pixel pixel(rng, dopfact, rbin);

                // Initialize LLH to middle of input DEM and average height
                Vec3 llh = demInterp.midLonLat();

                // Perform rdr->geo iterations
                int geostat = rdr2geo(
                    pixel, TCNbasis, pos, vel, _ellipsoid, demInterp, llh,
                    _radarGrid.lookSide(), _threshold, _numiter, _extraiter);
                totalconv += geostat;

                // Save data in output arrays
                _setOutputTopoLayers(llh, layers, blockLine, pixel, pos, vel, TCNbasis, demInterp);

            } // end OMP for loop pixels in block
        } // end for loop lines in block
        printf("\rTopo progress (block %d/%d): 100%%\n",
               (int) block + 1, (int) nBlocks), fflush(stdout);

        // Compute layover/shadow masks for the block
        if (_computeMask) {
            setLayoverShadow(layers, demInterp, satPosition, block, nBlocks);
        }

        // Write out block of data for all topo layers
        layers.writeData(0, lineStart);

    } // end for loop blocks

    // Print out convergence statistics
    info << "Total convergence: " << totalconv << " out of "
         << _radarGrid.size() << pyre::journal::endl;

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
        timerEnd - timerStart).count();
    info << "Elapsed processing time: " << elapsed << " sec"
         << pyre::journal::newline;
}

void isce3::geometry::Topo::topo(DEMInterpolator& demInterp,
                                TopoLayers& layers) {
    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.geometry.Topo");
    pyre::journal::info_t info("isce.geometry.Topo");

    // Create and start a timer
    auto timerStart = std::chrono::steady_clock::now();

    // Compute number of blocks needed to process image
    size_t nBlocks = _radarGrid.length() / _linesPerBlock;
    if ((_radarGrid.length() % _linesPerBlock) != 0)
        nBlocks += 1;

    // Cache range bounds for diagnostics
    const double startingRange = _radarGrid.startingRange();
    const double endingRange = _radarGrid.endingRange();
    const double midRange = _radarGrid.midRange();

    // Loop over blocks
    size_t totalconv = 0;
    for (size_t block = 0; block < nBlocks; ++block) {

        // Get block extents
        size_t lineStart, blockLength;
        lineStart = block * _linesPerBlock;
        if (block == (nBlocks - 1)) {
            blockLength = _radarGrid.length() - lineStart;
        } else {
            blockLength = std::min(_linesPerBlock, _radarGrid.length());
        }

        // Diagnostics
        const double tblock = _radarGrid.sensingTime(lineStart);
        info << "Processing block: " << block + 1 << " " << pyre::journal::newline
             << "  - line start: " << lineStart << pyre::journal::newline
             << "  - line end  : " << lineStart + blockLength
             << pyre::journal::newline << "  - dopplers near mid far: "
             << _doppler.eval(tblock, startingRange) << " "
             << _doppler.eval(tblock, midRange) << " "
             << _doppler.eval(tblock, endingRange) << " "
             << pyre::journal::endl;

        // Compute max and mean DEM height for the subset
        float demmin, demmax, dem_avg;
        demInterp.computeMinMaxMeanHeight(demmin, demmax, dem_avg);
        // Reset reference height for DEMInterpolator
        demInterp.refHeight(dem_avg);

        // Reset output block sizes in layers
        layers.setBlockSize(blockLength, _radarGrid.width());

        // Allocate vector for storing satellite position for each line
        std::vector<Vec3> satPosition(blockLength);

        // For each line in block
        double tline;
        for (size_t blockLine = 0; blockLine < blockLength; ++blockLine) {

            if (blockLine % std::max((int) (blockLength / 100), 1) == 0)
                    printf("\rTopo progress (block %d/%d): %d%%",
                           (int) block + 1, (int) nBlocks,
                           (int) (blockLine * 1e2 / blockLength)),
                           fflush(stdout);

            // Global line index
            size_t line = lineStart + blockLine;

            // Initialize orbital data for this azimuth line
            Basis TCNbasis;
            Vec3 pos, vel;
            _initAzimuthLine(line, tline, pos, vel, TCNbasis);
            satPosition[blockLine] = pos;

            // Compute velocity magnitude
            const double satVmag = vel.norm();

            // For each slant range bin
            #pragma omp parallel for reduction(+ : totalconv)
            for (size_t rbin = 0; rbin < _radarGrid.width(); ++rbin) {

                // Get current slant range
                const double rng = _radarGrid.slantRange(rbin);

                // Get current Doppler value
                const double dopfact = (0.5 * _radarGrid.wavelength() *
                                        (_doppler.eval(tline, rng) / satVmag)) *
                                       rng;

                // Store slant range bin data in Pixel
                Pixel pixel(rng, dopfact, rbin);

                // Initialize LLH to middle of input DEM and average height
                Vec3 llh = demInterp.midLonLat();

                // Perform rdr->geo iterations
                int geostat = rdr2geo(pixel, TCNbasis, pos, vel, _ellipsoid,
                                      demInterp, llh, _radarGrid.lookSide(),
                                      _threshold, _numiter, _extraiter);
                totalconv += geostat;

                // Save data in output arrays
                _setOutputTopoLayers(llh, layers, blockLine, pixel, pos, vel,
                                     TCNbasis, demInterp);

            } // end OMP for loop pixels in block
        } // end for loop lines in block
        printf("\rTopo progress (block %d/%d): 100%%\n",
               (int) block + 1, (int) nBlocks), fflush(stdout);

        // Compute layover/shadow masks for the block
        if (_computeMask) {
            setLayoverShadow(layers, demInterp, satPosition, block, nBlocks);
        }

        // Write out block of data for all topo layers
        layers.writeData(0, lineStart);

    } // end for loop blocks

    // Print out convergence statistics
    info << "Total convergence: " << totalconv << " out of "
         << _radarGrid.size() << pyre::journal::endl;

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed =
            1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
                             timerEnd - timerStart)
                             .count();
    info << "Elapsed processing time: " << elapsed << " sec"
         << pyre::journal::newline;
}


void isce3::geometry::Topo::topo(isce3::io::Raster& demRaster,
                                const std::string& outdir) {
    _topo(demRaster, outdir);
}


void isce3::geometry::Topo::topo(
        isce3::io::Raster& demRaster, isce3::io::Raster* xRaster,
        isce3::io::Raster* yRaster, isce3::io::Raster* heightRaster,
        isce3::io::Raster* incRaster, isce3::io::Raster* hdgRaster,
        isce3::io::Raster* localIncRaster, isce3::io::Raster* localPsiRaster,
        isce3::io::Raster* simRaster, isce3::io::Raster* maskRaster) {
    _topo(demRaster, xRaster, yRaster, heightRaster, incRaster, hdgRaster,
          localIncRaster, localPsiRaster, simRaster, maskRaster);
}

void isce3::geometry::Topo::topo(isce3::geometry::DEMInterpolator& demInterp,
                                const std::string& outdir) {
    _topo(demInterp, outdir);
}

void isce3::geometry::Topo::topo(
        isce3::geometry::DEMInterpolator& demInterp, isce3::io::Raster* xRaster,
        isce3::io::Raster* yRaster, isce3::io::Raster* heightRaster,
        isce3::io::Raster* incRaster, isce3::io::Raster* hdgRaster,
        isce3::io::Raster* localIncRaster, isce3::io::Raster* localPsiRaster,
        isce3::io::Raster* simRaster, isce3::io::Raster* maskRaster) {
    _topo(demInterp, xRaster, yRaster, heightRaster, incRaster, hdgRaster,
          localIncRaster, localPsiRaster, simRaster, maskRaster);
}

void isce3::geometry::Topo::
_initAzimuthLine(size_t line, double& tline, Vec3& pos, Vec3& vel, Basis& TCNbasis)
{
    // Get satellite azimuth time
    tline = _radarGrid.sensingTime(line);

    // Get state vector
    _orbit.interpolate(&pos, &vel, tline,
                       isce3::core::OrbitInterpBorderMode::FillNaN);

    // Get geocentric TCN basis using satellite basis
    TCNbasis = Basis(pos, vel);
}

// Get DEM bounds using first/last azimuth line and slant range bin
void isce3::geometry::Topo::
computeDEMBounds(Raster & demRaster, DEMInterpolator & demInterp, size_t lineOffset,
                 size_t blockLength)
{
    // Initialize journal
    pyre::journal::warning_t warning("isce.core.Topo");

    // Initialize geographic bounds
    double minX = 1.0e64;
    double maxX = -1.0e64;
    double minY = 1.0e64;
    double maxY = -1.0e64;

    // Skip factors along azimuth and range
    const int askip = std::max((int) blockLength / 10, 1);
    const int rskip = _radarGrid.width() / 10;

    //Construct projection base with DEM's epsg code
    int epsgcode = demRaster.getEPSG();

    isce3::core::ProjectionBase * proj = isce3::core::createProj(epsgcode);

    // Construct vectors of range/azimuth indices traversing the perimeter of the radar frame

    // Top edge
    std::vector<int> azInd, rgInd;
    for (int j = 0; j < _radarGrid.width(); j += rskip) {
        azInd.push_back(0);
        rgInd.push_back(j);
    }

    // Right edge
    for (int i = 0; i < blockLength; i += askip) {
        azInd.push_back(i);
        rgInd.push_back(_radarGrid.width());
    }

    // Bottom edge
    for (int j = _radarGrid.width(); j > 0; j -= rskip) {
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
        size_t lineIndex = lineOffset + azInd[i];

         // Initialize orbit data for this azimuth line
        Vec3 pos, vel;
        Basis TCNbasis;
        _initAzimuthLine(lineIndex, tline, pos, vel, TCNbasis);
        // Compute satellite velocity and height
        const double satVmag = vel.norm();
        const Vec3 satLLH = _ellipsoid.xyzToLonLat(pos);

        // Get proper slant range and Doppler factor
        const size_t rbin = rgInd[i];
        double rng = _radarGrid.slantRange(rbin);
        double dopfact = (0.5 * _radarGrid.wavelength() * (_doppler.eval(tline, rng)
                        / satVmag)) * rng;
        // Store in Pixel object
        Pixel pixel(rng, dopfact, rbin);

        // Run topo for one iteration for two different heights
        Vec3 llh {0., 0., 0.};

        std::array<double, 2> testHeights = {_minH, _maxH};
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
                rdr2geo(pixel, TCNbasis, pos, vel, _ellipsoid, constDEM, llh,
                        _radarGrid.lookSide(), _threshold, 1, 0);
            }

            Vec3 enu = {0., 0., 0.};
            proj->forward(llh, enu);

            // Update bounds
            minY = std::min(minY, enu[1]);
            maxY = std::max(maxY, enu[1]);
            minX = std::min(minX, enu[0]);
            maxX = std::max(maxX, enu[0]);
        }
    }

    //Convert margin to meters it not LonLat
    double margin = (epsgcode == 4326)? _margin : isce3::core::decimaldeg2meters(_margin);

    // Account for margins
    minX -= margin;
    maxX += margin;
    minY -= margin;
    maxY += margin;

    // Extract DEM subset
    demInterp.loadDEM(demRaster, minX, maxX, minY, maxY);

    demInterp.declare();
}

void isce3::geometry::Topo::
_setOutputTopoLayers(Vec3 & targetLLH, TopoLayers & layers, size_t line,
                     Pixel & pixel, Vec3& pos, Vec3& vel, Basis & TCNbasis,
                     DEMInterpolator & demInterp)
{
    const double degrees = 180.0 / M_PI;

    // Unpack the range pixel data
    const size_t bin = pixel.bin();

    // Convert lat/lon values to output coordinate system
    Vec3 xyzOut;
    _proj->forward(targetLLH, xyzOut);
    const double x = xyzOut[0];
    const double y = xyzOut[1];

    // Set outputs
    layers.x(line, bin, x);
    layers.y(line, bin, y);
    layers.z(line, bin, targetLLH[2]);

    // Skip other computations if their rasters aren't set
    if (!layers.hasIncRaster() &&
            !layers.hasHdgRaster() &&
            !layers.hasLocalIncRaster() &&
            !layers.hasLocalPsiRaster() &&
            !layers.hasSimRaster() &&
            !layers.hasMaskRaster()) {
        return;
    }

    // Convert llh->xyz for ground point
    const Vec3 targetXYZ = _ellipsoid.lonLatToXyz(targetLLH);

    // Compute vector from satellite to ground point
    const Vec3 satToGround = targetXYZ - pos;

    // Compute cross-track range
    if (_radarGrid.lookSide() == isce3::core::LookSide::Right) {
        layers.crossTrack(line, bin, satToGround.dot(TCNbasis.x1()));
    } else {
        layers.crossTrack(line, bin, -satToGround.dot(TCNbasis.x1()));
    }

    // Computation in ENU coordinates around target
    const Mat3 xyz2enu = Mat3::xyzToEnu(targetLLH[1], targetLLH[0]);
    const Vec3 enu = xyz2enu.dot(satToGround);
    const double cosalpha = std::abs(enu[2]) / enu.norm();

    // Incidence angle
    layers.inc(line, bin, std::acos(cosalpha) * degrees);

    // Skip other computations if their rasters aren't set
    if (!layers.hasHdgRaster() &&
            !layers.hasLocalIncRaster() &&
            !layers.hasLocalPsiRaster() &&
            !layers.hasSimRaster()) {
        return;
    }

    // Heading considering zero-Doppler grid and anti-clock. ref. starting from the East
    double heading;
    if (_radarGrid.lookSide() == isce3::core::LookSide::Left) {
        heading = (std::atan2(enu[1], enu[0]) - (0.5*M_PI)) * degrees;
    } else {
        heading = (std::atan2(enu[1], enu[0]) + (0.5*M_PI)) * degrees;
    }
    if (heading > 180) {
        heading -= 360;
    } else if (heading < -180) {
        heading += 360;
    }
    layers.hdg(line, bin, heading);

    // Project output coordinates to DEM coordinates
    auto input_coords_llh = _proj->inverse({x, y, targetLLH[2]});
    Vec3 dem_vect = demInterp.proj()->forward(input_coords_llh);

    // East-west slope using central difference
    double aa = demInterp.interpolateXY(dem_vect[0] - demInterp.deltaX(), dem_vect[1]);
    double bb = demInterp.interpolateXY(dem_vect[0] + demInterp.deltaX(), dem_vect[1]);

    Vec3 dem_vect_p_dx = {dem_vect[0] + demInterp.deltaX(), dem_vect[1], dem_vect[2]};
    Vec3 dem_vect_m_dx = {dem_vect[0] - demInterp.deltaX(), dem_vect[1], dem_vect[2]};
    Vec3 input_coords_llh_p_dx, input_coords_llh_m_dx;
    demInterp.proj()->inverse(dem_vect_p_dx, input_coords_llh_p_dx);
    demInterp.proj()->inverse(dem_vect_m_dx, input_coords_llh_m_dx);
    const Vec3 input_coords_xyz_p_dx = _ellipsoid.lonLatToXyz(input_coords_llh_p_dx);
    const Vec3 input_coords_xyz_m_dx = _ellipsoid.lonLatToXyz(input_coords_llh_m_dx);
    double dx = (input_coords_xyz_p_dx - input_coords_xyz_m_dx).norm();

    // Compute east-west slope using plus-minus sign from deltaX() (usually positive)
    double alpha = std::copysign((bb - aa) / dx, (bb - aa) * demInterp.deltaX());

    // North-south slope using central difference
    aa = demInterp.interpolateXY(dem_vect[0], dem_vect[1] - demInterp.deltaY());
    bb = demInterp.interpolateXY(dem_vect[0], dem_vect[1] + demInterp.deltaY());

    Vec3 dem_vect_p_dy = {dem_vect[0], dem_vect[1] + demInterp.deltaY(), dem_vect[2]};
    Vec3 dem_vect_m_dy = {dem_vect[0], dem_vect[1] - demInterp.deltaY(), dem_vect[2]};
    Vec3 input_coords_llh_p_dy, input_coords_llh_m_dy;
    demInterp.proj()->inverse(dem_vect_p_dy, input_coords_llh_p_dy);
    demInterp.proj()->inverse(dem_vect_m_dy, input_coords_llh_m_dy);
    const Vec3 input_coords_xyz_p_dy = _ellipsoid.lonLatToXyz(input_coords_llh_p_dy);
    const Vec3 input_coords_xyz_m_dy = _ellipsoid.lonLatToXyz(input_coords_llh_m_dy);
    double dy = (input_coords_xyz_p_dy - input_coords_xyz_m_dy).norm();

    // Compute north-south slope using plus-minus sign from deltaY() (usually negative)
    double beta = std::copysign((bb - aa) / dy, (bb - aa) * demInterp.deltaY());

    // Compute local incidence angle
    const Vec3 enunorm = enu.normalized();
    const Vec3 slopevec {alpha, beta, -1.};
    const double costheta = enunorm.dot(slopevec) / slopevec.norm();
    layers.localInc(line, bin, std::acos(costheta)*degrees);

    // Compute amplitude simulation
    double sintheta = std::sqrt(1.0 - (costheta * costheta));
    bb = sintheta + 0.1 * costheta;
    layers.sim(line, bin, std::log10(std::abs(0.01 * costheta / (bb * bb * bb))));

    // Calculate psi angle between image plane and local slope
    Vec3 n_imghat = satToGround.cross(vel).normalized();
    if (_radarGrid.lookSide() == isce3::core::LookSide::Left) {
        n_imghat *= -1.0;
    }
    Vec3 n_img_enu = xyz2enu.dot(n_imghat);
    const Vec3 n_trg_enu = -slopevec;
    const double cospsi = n_trg_enu.dot(n_img_enu)
          / (n_trg_enu.norm() * n_img_enu.norm());
    layers.localPsi(line, bin, std::acos(cospsi) * degrees);
}

void isce3::geometry::Topo::
setLayoverShadow(TopoLayers& layers, DEMInterpolator& demInterp,
                 std::vector<Vec3>& satPosition, size_t block,
                 size_t n_blocks)
{
    // Cache the width of the block
    const int width = layers.width();
    // Compute layover on oversampled grid
    const int gridWidth = 2 * width;

    // Allocate working valarrays
    std::valarray<double> x(width), y(width), ctrack(width), ctrackGrid(gridWidth);
    std::valarray<double> slantRangeGrid(gridWidth);
    std::valarray<short> maskGrid(gridWidth);

    // Initialize mask to zero for this block
    layers.mask() = 0;

    // Prepare function getDemCoords() to interpolate DEM
    std::function<Vec3(double, double,
                       const isce3::geometry::DEMInterpolator&,
                       isce3::core::ProjectionBase*)> getDemCoords;

    if (_epsgOut == demInterp.epsgCode()) {
        getDemCoords = isce3::geometry::getDemCoordsSameEpsg;
    } else {
        getDemCoords = isce3::geometry::getDemCoordsDiffEpsg;
    }

    long long num_lines_done = 0;

    // Loop over lines in block
    #pragma omp parallel for firstprivate(x, y, ctrack, ctrackGrid, \
                                          slantRangeGrid, maskGrid) \
                             shared(num_lines_done)
    for (size_t line = 0; line < layers.length(); ++line) {

        // Cache satellite position for this line
        const Vec3& xyzsat = satPosition[line];

        // Copy cross-track, x, and y values for the line
        for (int i = 0; i < width; ++i) {
            ctrack[i] = layers.crossTrack(line, i);
            x[i] = layers.x(line, i);
            y[i] = layers.y(line, i);
        }

        // Sort ctrack, x, and y by values in ctrack
        isce3::core::insertionSort(ctrack, x, y);

        // Create regular grid for cross-track values
        const double cmin = ctrack.min();// - demInterp.maxHeight();
        const double cmax = ctrack.max();// + demInterp.maxHeight();
        isce3::core::linspace<double>(cmin, cmax, ctrackGrid);

        // Interpolate DEM to regular cross-track grid
        for (int i = 0; i < gridWidth; ++i) {

            // Compute nearest ctrack index for current ctrackGrid value
            const double crossTrack = ctrackGrid[i];
            int k = isce3::core::binarySearch(ctrack, crossTrack);
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
            Vec3 demXYZ = getDemCoords(x_grid, y_grid, demInterp, _proj);

            // Convert DEM XYZ to ECEF XYZ
            Vec3 llh, xyz, satToGround;
            demInterp.proj()->inverse(demXYZ, llh);
            _ellipsoid.lonLatToXyz(llh, xyz);

            // Compute and save slant range
            satToGround = xyz - xyzsat;
            slantRangeGrid[i] = satToGround.norm();
        }

        // Now sort cross-track grid in terms of slant range grid
        isce3::core::insertionSort(slantRangeGrid, ctrackGrid);

        // Traverse from near range to far range on original spacing for shadow detection
        double minIncAngle = layers.inc(line, 0);
        for (int i = 1; i < width; ++i) {
            const double inc = layers.inc(line, i);
            // Test shadow
            if (inc <= minIncAngle) {
                layers.mask(line, i, isce3::core::SHADOW_VALUE);
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
                layers.mask(line, i, isce3::core::SHADOW_VALUE);
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
                maskGrid[i] = isce3::core::LAYOVER_VALUE;
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
                maskGrid[i] = isce3::core::LAYOVER_VALUE;
            } else {
                maxCrossTrack = crossTrack;
            }
        }

        // Resample maskGrid to original spacing
        for (int i = 0; i < gridWidth; ++i) {
            if (maskGrid[i] > 0) {

                const long slant_range_index = static_cast<long>(
                    std::floor(
                        (slantRangeGrid[i] - _radarGrid.startingRange()) /
                        _radarGrid.rangePixelSpacing()));

                // If out of bounds, escape
                if (slant_range_index < 0 || slant_range_index >= width) {
                    continue;
                }

                // Otherwise, update it
                const short mask_value = layers.mask(line, slant_range_index);

                /*
                `mask_grid[i]` can be 0 or 0b0010 (LAYOVER_VALUE: decimal value 2)
                `mask_value` can be 0 or 0b0001 (SHADOW_VALUE: decimal value 1)
                             or 0b0010 (LAYOVER_VALUE: decimal value 2)

                The operator "|" represents the bitwise-or of `mask_value` and
                `mask_grid[i]`:

                `mask_grid[i]`   ,      `mask_value`    ,     `new_mask_value`
                      0          ,  0                   ,  0
                      0          ,  1 (SHADOW)          ,  1 (SHADOW)
                      0          ,  2 (LAYOVER)         ,  2 (LAYOVER)
                      0          ,  3 (LAYOVER & SHADOW),  3 (LAYOVER & SHADOW)
                      1 (SHADOW) ,  0                   ,  1 (SHADOW)
                      1 (SHADOW) ,  1 (SHADOW)          ,  1 (SHADOW)
                      1 (SHADOW) ,  2 (LAYOVER)         ,  3 (LAYOVER & SHADOW)
                      1 (SHADOW) ,  3 (LAYOVER & SHADOW),  3 (LAYOVER & SHADOW)
                */
                const short new_mask_value = mask_value | maskGrid[i];
                if (mask_value != new_mask_value) {
                    layers.mask(line, slant_range_index, new_mask_value);
                }
            }
        }
    
    _Pragma("omp atomic")
        num_lines_done++;
    if (line % std::max((int) (layers.length() / 100), 1) == 0)
        _Pragma("omp critical")
            printf("\rLayover/shadow mask progress (block %d/%d): %d%%",
                   (int) block + 1, (int) n_blocks,
                   (int) (num_lines_done * 1e2 / layers.length())),
                   fflush(stdout);
                        
    } // end loop lines

printf("\rLayover/shadow mask progress (block %d/%d): 100%%\n",
       (int) block + 1, (int) n_blocks), fflush(stdout);
}

