// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#include <chrono>
#include "Topo.h"
#include "gpuTopo.h"

// pull in some isce namespaces
using isce::core::Ellipsoid;
using isce::core::Orbit;
using isce::core::Poly2d;
using isce::product::ImageMode;
using isce::io::Raster;
using isce::geometry::DEMInterpolator;
using isce::geometry::TopoLayers;

// Main topo driver
void isce::cuda::geometry::Topo::
topo(Raster & demRaster,
     const std::string outdir) {

    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.cuda.geometry.Topo");
    pyre::journal::info_t info("isce.cuda.geometry.Topo");

    // First check that variables have been initialized
    checkInitialization(info); 

    // Cache ISCE objects (use public interface of parent isce::geometry::Topo class)
    ImageMode mode = this->mode();
    Ellipsoid ellipsoid = this->ellipsoid();
    Orbit orbit = this->orbit();
    Poly2d doppler = this->doppler();

    { // Topo scope for creating output rasters

    // Create rasters for individual layers
    Raster xRaster = Raster(outdir + "/x.rdr", mode.width(), mode.length(), 1,
        GDT_Float64, "ISCE");
    Raster yRaster = Raster(outdir + "/y.rdr", mode.width(), mode.length(), 1,
        GDT_Float64, "ISCE");
    Raster heightRaster = Raster(outdir + "/z.rdr", mode.width(), mode.length(), 1,
        GDT_Float64, "ISCE");
    Raster incRaster = Raster(outdir + "/inc.rdr", mode.width(), mode.length(), 1,
        GDT_Float32, "ISCE");
    Raster hdgRaster = Raster(outdir + "/hdg.rdr", mode.width(), mode.length(), 1,
        GDT_Float32, "ISCE");
    Raster localIncRaster = Raster(outdir + "/localInc.rdr", mode.width(), mode.length(), 1,
        GDT_Float32, "ISCE");
    Raster localPsiRaster = Raster(outdir + "/localPsi.rdr", mode.width(), mode.length(), 1,
        GDT_Float32, "ISCE");
    Raster simRaster = Raster(outdir + "/simamp.rdr", mode.width(), mode.length(), 1,
        GDT_Float32, "ISCE");

    // Create and start a timer
    auto timerStart = std::chrono::steady_clock::now();

    // Create a DEM interpolator
    DEMInterpolator demInterp(-500.0);

    // Compute number of blocks needed to process image
    size_t nBlocks = mode.length() / _linesPerBlock;
    if ((mode.length() % _linesPerBlock) != 0)
        nBlocks += 1;

    // Loop over blocks
    size_t totalconv = 0;
    for (size_t block = 0; block < nBlocks; ++block) {

        // Get block extents
        size_t lineStart, blockLength;
        lineStart = block * _linesPerBlock;
        if (block == (nBlocks - 1)) {
            blockLength = mode.length() - lineStart;
        } else {
            blockLength = _linesPerBlock;
        }

        // Diagnostics
        info << "Processing block: " << block << " " << pyre::journal::newline
             << "  - line start: " << lineStart << pyre::journal::newline
             << "  - line end  : " << lineStart + blockLength << pyre::journal::newline
             << "  - dopplers near mid far: "
             << doppler.eval(0, 0) << " "
             << doppler.eval(0, (mode.width() / 2) - 1) << " "
             << doppler.eval(0, mode.width() - 1) << " "
             << pyre::journal::endl;

        // Load DEM subset for SLC image block
        computeDEMBounds(demRaster, demInterp, lineStart, blockLength);

        // Compute max and mean DEM height for the subset
        float demmax, dem_avg;
        demInterp.computeHeightStats(demmax, dem_avg, info);
        // Reset reference height using mean
        demInterp.refHeight(dem_avg);

        // Output layers for block
        TopoLayers layers(blockLength, mode.width());

        // Run Topo on the GPU for this block
        isce::cuda::geometry::runGPUTopo(
            ellipsoid, orbit, doppler, mode, demInterp, layers, lineStart, this->lookSide(),
            this->epsgOut(), this->threshold(), this->numiter(), this->extraiter()
        );
        
        // Write out block of data for every product
        xRaster.setBlock(layers.x(), 0, lineStart, mode.width(), blockLength);
        yRaster.setBlock(layers.y(), 0, lineStart, mode.width(), blockLength);
        heightRaster.setBlock(layers.z(), 0, lineStart, mode.width(), blockLength);
        incRaster.setBlock(layers.inc(), 0, lineStart, mode.width(), blockLength);
        hdgRaster.setBlock(layers.hdg(), 0, lineStart, mode.width(), blockLength);
        localIncRaster.setBlock(layers.localInc(), 0, lineStart, mode.width(), blockLength);
        localPsiRaster.setBlock(layers.localPsi(), 0, lineStart, mode.width(), blockLength);
        simRaster.setBlock(layers.sim(), 0, lineStart, mode.width(), blockLength);

    } // end for loop blocks

    // Print out convergence statistics
    info << "Total convergence: " << totalconv << " out of "
         << (mode.width() * mode.length()) << pyre::journal::endl;

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
    vrt.setEPSG(this->epsgOut());

}

// end of file
