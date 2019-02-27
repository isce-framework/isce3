// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#include <chrono>
#include "Topo.h"
#include "utilities.h"
#include "gpuTopo.h"

// pull in some isce namespaces
using isce::core::Ellipsoid;
using isce::core::Orbit;
using isce::core::LUT1d;
using isce::core::cartesian_t;
using isce::io::Raster;
using isce::product::RadarGridParameters;
using isce::geometry::DEMInterpolator;
using isce::geometry::TopoLayers;

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
 * <li> mask.rdr - Layover and shadow image.
 * </ul>*/
void isce::cuda::geometry::Topo::
topo(Raster & demRaster,
     const std::string outdir) {

    { // Topo scope for creating output rasters

    // Initialize a TopoLayers object to handle block data and raster data
    TopoLayers layers;

    // Create rasters for individual layers (provide output raster sizes)
    const RadarGridParameters & radarGrid = this->radarGridParameters();
    layers.initRasters(outdir, radarGrid.width(), radarGrid.length(),
                       this->computeMask());

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
    if (this->computeMask()) {
        rasterTopoVec.push_back(Raster(outdir + "/mask.rdr" ));
    };

    Raster vrt = Raster(outdir + "/topo.vrt", rasterTopoVec );
    // Set its EPSG code
    vrt.setEPSG(this->epsgOut());
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
void isce::cuda::geometry::Topo::
topo(Raster & demRaster, Raster & xRaster, Raster & yRaster, Raster & heightRaster,
     Raster & incRaster, Raster & hdgRaster, Raster & localIncRaster, Raster & localPsiRaster,
     Raster & simRaster, Raster & maskRaster) {

    // Initialize a TopoLayers object to handle block data and raster data
    TopoLayers layers;

    // Create rasters for individual layers (provide output raster sizes)
    layers.setRasters(xRaster, yRaster, heightRaster, incRaster, hdgRaster, localIncRaster,
                      localPsiRaster, simRaster, maskRaster);
    // Indicate a mask raster has been provided for writing
    this->computeMask(true);

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
  * @param[in] simRaster output raster for simulated amplitude image. 
  * @param[in] maskRaster output raster for layover/shadow mask. */
void isce::cuda::geometry::Topo::
topo(Raster & demRaster, Raster & xRaster, Raster & yRaster, Raster & heightRaster,
     Raster & incRaster, Raster & hdgRaster, Raster & localIncRaster, Raster & localPsiRaster,
     Raster & simRaster) {

    // Initialize a TopoLayers object to handle block data and raster data
    TopoLayers layers;

    // Create rasters for individual layers (provide output raster sizes)
    layers.setRasters(xRaster, yRaster, heightRaster, incRaster, hdgRaster, localIncRaster,
                      localPsiRaster, simRaster);
    // Indicate no mask raster has been provided for writing
    this->computeMask(false);

    // Call topo with layers
    topo(demRaster, layers);
}

/** @param[in] demRaster input DEM raster
  * @param[in] layers TopoLayers object for storing and writing results
  */
void isce::cuda::geometry::Topo::
topo(Raster & demRaster, TopoLayers & layers) {

    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.cuda.geometry.Topo");
    pyre::journal::info_t info("isce.cuda.geometry.Topo");

    // Cache ISCE objects (use public interface of parent isce::geometry::Topo class)
    const Ellipsoid & ellipsoid = this->ellipsoid();
    const Orbit & orbit = this->orbit();
    const LUT1d<double> doppler(this->doppler());
    const RadarGridParameters & radarGrid = this->radarGridParameters();

    // Create and start a timer
    auto timerStart = std::chrono::steady_clock::now();

    // Create a DEM interpolator
    DEMInterpolator demInterp(-500.0, this->demMethod());

    // Compute number of lines per block
    computeLinesPerBlock(demRaster);

    // Compute number of blocks needed to process image
    size_t nBlocks = radarGrid.length() / _linesPerBlock;
    if ((radarGrid.length() % _linesPerBlock) != 0)
        nBlocks += 1;

    // Loop over blocks
    unsigned int totalconv = 0;
    for (size_t block = 0; block < nBlocks; ++block) {

        // Get block extents
        size_t lineStart, blockLength;
        lineStart = block * _linesPerBlock;
        if (block == (nBlocks - 1)) {
            blockLength = radarGrid.length() - lineStart;
        } else {
            blockLength = _linesPerBlock;
        }

        // Diagnostics
        info << "Processing block: " << block << " " << pyre::journal::newline
             << "  - line start: " << lineStart << pyre::journal::newline
             << "  - line end  : " << lineStart + blockLength << pyre::journal::newline
             << "  - dopplers near mid far: "
             << doppler.values()[0] << " "
             << doppler.values()[doppler.size() / 2] << " "
             << doppler.values()[doppler.size() - 1] << " "
             << pyre::journal::endl;

        // Load DEM subset for SLC image block
        computeDEMBounds(demRaster, demInterp, lineStart, blockLength);

        // Compute max and mean DEM height for the subset
        float demmax, dem_avg;
        demInterp.computeHeightStats(demmax, dem_avg, info);
        // Reset reference height using mean
        demInterp.refHeight(dem_avg);

        // Set output block sizes in layers
        layers.setBlockSize(blockLength, radarGrid.width());

        // Run Topo on the GPU for this block
        isce::cuda::geometry::runGPUTopo(
            ellipsoid, orbit, doppler, demInterp, layers, lineStart, this->lookSide(),
            this->epsgOut(), radarGrid.numberAzimuthLooks(), radarGrid.sensingStart(),
            radarGrid.wavelength(), radarGrid.prf(), radarGrid.startingRange(),
            radarGrid.rangePixelSpacing(), this->threshold(), this->numiter(), this->extraiter(),
            totalconv
        );

        // Compute layover/shadow masks for the block
        if (this->computeMask()) {
            _setLayoverShadowWithOrbit(orbit, layers, demInterp, lineStart);
        }

        // Write out block of data for all topo layers
        layers.writeData(0, lineStart);

    } // end for loop blocks

    // Print out convergence statistics
    info << "Total convergence: " << totalconv << " out of "
         << (radarGrid.size()) << pyre::journal::endl;

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
        timerEnd - timerStart).count();
    info << "Elapsed processing time: " << elapsed << " sec"
         << pyre::journal::newline;
}

// Compute number of lines per block dynamically from GPU memory
void isce::cuda::geometry::Topo::
computeLinesPerBlock(isce::io::Raster & demRaster) {

    // Compute GPU memory
    const size_t nGPUBytes = getDeviceMem();

    // Assume entire DEM passed to GPU
    const size_t nDEMBytes = demRaster.width() * demRaster.length() * sizeof(float);

    // 2 GB buffer for safeguard for large enough devices (> 6 GB)
    size_t gpuBuffer;
    if (nGPUBytes > 6e9) {
        gpuBuffer = 2e9;
    } else {
        // Else use 500 MB buffer
        gpuBuffer = 500e6;
    }

    // Compute pixels per Block (4 double and 5 float output topo layers)
    size_t pixelsPerBlock = (nGPUBytes - nDEMBytes - gpuBuffer) /
                            (4 * sizeof(double) + 5 * sizeof(float));
    // Round down to nearest 10 million pixels
    pixelsPerBlock = (pixelsPerBlock / 10000000) * 10000000;

    // Compute number of lines per block
    _linesPerBlock = pixelsPerBlock / this->radarGridParameters().width();
    // Round down to nearest 500 lines
    _linesPerBlock = (_linesPerBlock / 500) * 500;
}

// Compute layover and shadow masks
void isce::cuda::geometry::Topo::
_setLayoverShadowWithOrbit(const isce::core::Orbit & orbit,
                           TopoLayers & layers,
                           DEMInterpolator & demInterp,
                           size_t lineStart) {
    
    // Create vector of satellite positions for each line in block
    std::vector<cartesian_t> satPosition(layers.length());
    for (size_t i = 0; i < layers.length(); ++i) {

        // Get satellite azimuth time
        const double tline = this->radarGridParameters().sensingTime(i + lineStart);

        // Get state vector
        cartesian_t xyzsat, velsat;
        orbit.interpolateWGS84Orbit(tline, xyzsat, velsat);
        satPosition[i] = xyzsat;
    }
        
    // Call standard layover/shadow mask generation function
    this->setLayoverShadow(layers, demInterp, satPosition);
}

// end of file
