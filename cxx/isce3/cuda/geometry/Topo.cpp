// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#include "Topo.h"

#include <chrono>
#include "utilities.h"
#include "gpuTopo.h"
#include <isce3/core/LUT1d.h>
#include <isce3/core/Orbit.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/TopoLayers.h>

// pull in some isce namespaces
using isce3::core::Ellipsoid;
using isce3::core::Orbit;
using isce3::core::OrbitInterpBorderMode;
using isce3::core::LUT1d;
using isce3::core::cartesian_t;
using isce3::io::Raster;
using isce3::product::RadarGridParameters;
using isce3::geometry::DEMInterpolator;
using isce3::geometry::TopoLayers;

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
 * <li> layoverShadowMask.rdr - Layover and shadow image.
 * <li> los_east.rdr - East component of ground to satellite unit vector
 * <li> los_north.rdr - North component of ground to satellite unit vector
 * </ul>*/
void isce3::cuda::geometry::Topo::
topo(Raster & demRaster,
     const std::string & outdir) {

    { // Topo scope for creating output rasters

    // Create rasters for individual layers (provide output raster sizes)
    const RadarGridParameters & radarGrid = this->radarGridParameters();

    // Initialize a TopoLayers object to handle block data and raster data
    TopoLayers layers(outdir, radarGrid.width(), radarGrid.length(),
            linesPerBlock(), this->computeMask());

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
        Raster(outdir + "/simamp.rdr" ),
        Raster(outdir + "/los_east.rdr" ),
        Raster(outdir + "/los_north.rdr" )
    };

    // Add optional mask raster
    if (this->computeMask()) {
        rasterTopoVec.push_back(Raster(outdir + "/layoverShadowMask.rdr" ));
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
  * @param[in] maskRaster output raster for layover/shadow mask.
  * @param[in] losEastRaster output for east component of ground to satellite LOS unit vector
  * @param[in] losNorthRaster output for north component of ground to satellite LOS unit vector
 */
void isce3::cuda::geometry::Topo::
topo(Raster & demRaster, Raster * xRaster, Raster * yRaster, Raster * heightRaster,
     Raster * incRaster, Raster * hdgRaster, Raster * localIncRaster, Raster * localPsiRaster,
     Raster * simRaster, Raster * maskRaster, Raster * losEastRaster,
     Raster * losNorthRaster) {

    // Initialize a TopoLayers object to handle block data and raster data
    // Create rasters for individual layers (provide output raster sizes)
    TopoLayers layers(linesPerBlock(), xRaster, yRaster, heightRaster, incRaster,
            hdgRaster, localIncRaster, localPsiRaster, simRaster,
            maskRaster, losEastRaster, losNorthRaster);

    // Set computeMask flag by pointer value
    this->computeMask(maskRaster != nullptr);

    // Call topo with layers
    topo(demRaster, layers);
}

/** @param[in] demRaster input DEM raster
  * @param[in] layers TopoLayers object for storing and writing results
  */
void isce3::cuda::geometry::Topo::
topo(Raster & demRaster, TopoLayers & layers) {

    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.cuda.geometry.Topo");
    pyre::journal::info_t info("isce.cuda.geometry.Topo");

    // Cache ISCE objects (use public interface of parent isce3::geometry::Topo class)
    const Ellipsoid & ellipsoid = this->ellipsoid();
    const Orbit & orbit = this->orbit();
    const LUT1d<double> doppler = isce3::core::avgLUT2dToLUT1d<double>(this->doppler());
    const RadarGridParameters & radarGrid = this->radarGridParameters();

    // Create and start a timer
    auto timerStart = std::chrono::steady_clock::now();

    // Create a DEM interpolator
    DEMInterpolator demInterp(-500.0, this->demMethod());

    // Compute number of blocks needed to process image
    size_t nBlocks = radarGrid.length() / linesPerBlock();
    if ((radarGrid.length() % linesPerBlock()) != 0)
        nBlocks += 1;

    // Cache near, mid, far ranges for diagnostics on Doppler
    const double nearRange = radarGrid.startingRange();
    const double farRange = radarGrid.endingRange();
    const double midRange = radarGrid.midRange();

    // Loop over blocks
    unsigned int totalconv = 0;
    for (size_t block = 0; block < nBlocks; ++block) {

        // Get block extents
        size_t lineStart, blockLength;
        lineStart = block * linesPerBlock();
        if (block == (nBlocks - 1)) {
            blockLength = radarGrid.length() - lineStart;
        } else {
            blockLength = linesPerBlock();
        }

        // Diagnostics
        info << "Processing block: " << block << " " << pyre::journal::newline
             << "  - line start: " << lineStart << pyre::journal::newline
             << "  - line end  : " << lineStart + blockLength << pyre::journal::newline
             << "  - dopplers near mid far: "
             << doppler.eval(nearRange) << " "
             << doppler.eval(midRange) << " "
             << doppler.eval(farRange) << " "
             << pyre::journal::endl;

        // Load DEM subset for SLC image block
        computeDEMBounds(demRaster, demInterp, lineStart, blockLength);

        // Compute max and mean DEM height for the subset
        float demmin, demmax, dem_avg;
        demInterp.computeMinMaxMeanHeight(demmin, demmax, dem_avg);
        // Reset reference height using mean
        demInterp.refHeight(dem_avg);

        // Reset output block sizes in layers
        layers.setBlockSize(blockLength, radarGrid.width());

        // Run Topo on the GPU for this block
        isce3::cuda::geometry::runGPUTopo(
            ellipsoid, orbit, doppler, demInterp, layers, lineStart, radarGrid.lookSide(),
            this->epsgOut(), radarGrid.sensingStart(),
            radarGrid.wavelength(), radarGrid.prf(), radarGrid.startingRange(),
            radarGrid.rangePixelSpacing(), this->threshold(), this->numiter(), this->extraiter(),
            totalconv
        );

        // Compute layover/shadow masks for the block
        if (this->computeMask()) {
            _setLayoverShadowWithOrbit(orbit, layers, demInterp, lineStart,
                                       block, nBlocks);
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

// Compute layover and shadow masks
void isce3::cuda::geometry::Topo::
_setLayoverShadowWithOrbit(const Orbit & orbit,
                           TopoLayers & layers,
                           DEMInterpolator & demInterp,
                           size_t lineStart,
                           size_t block,
                           size_t n_blocks) {

    // Create vector of satellite positions for each line in block
    std::vector<cartesian_t> satPosition(layers.length());
    for (size_t i = 0; i < layers.length(); ++i) {

        // Get satellite azimuth time
        const double tline = this->radarGridParameters().sensingTime(i + lineStart);

        // Get state vector
        cartesian_t xyzsat;
        orbit.interpolate(&xyzsat, nullptr, tline, OrbitInterpBorderMode::FillNaN);
        satPosition[i] = xyzsat;
    }

    // Call standard layover/shadow mask generation function
    this->setLayoverShadow(layers, demInterp, satPosition, block, n_blocks);
}

// end of file
