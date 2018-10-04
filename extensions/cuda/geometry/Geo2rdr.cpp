// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#include <chrono>
#include "Geo2rdr.h"
#include "utilities.h"
#include "gpuGeo2rdr.h"

// pull in some isce namespaces
using isce::core::Ellipsoid;
using isce::core::Orbit;
using isce::core::Poly2d;
using isce::core::DateTime;
using isce::product::ImageMode;
using isce::io::Raster;


// Run geo2rdr - main entrypoint
void isce::cuda::geometry::Geo2rdr::
geo2rdr(isce::io::Raster & topoRaster,
        const std::string & outdir,
        double azshift, double rgshift) {

    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.cuda.geometry.Geo2rdr");
    pyre::journal::info_t info("isce.cuda.geometry.Geo2rdr");

    // Cache the size of the DEM images
    const size_t demWidth = topoRaster.width();
    const size_t demLength = topoRaster.length();

    // Cache EPSG code for topo results
    const int topoEPSG = topoRaster.getEPSG();

    // Cache ISCE objects (use public interface of parent isce::geometry::Geo2rdr class)
    ImageMode mode = this->mode();
    Ellipsoid ellipsoid = this->ellipsoid();
    Orbit orbit = this->orbit();
    Poly2d doppler = this->doppler();
    DateTime sensingStart = this->sensingStart();
 
    // Create output rasters
    Raster rgoffRaster = Raster(outdir + "/range.off", demWidth, demLength, 1,
        GDT_Float32, "ISCE");
    Raster azoffRaster = Raster(outdir + "/azimuth.off", demWidth, demLength, 1,
        GDT_Float32, "ISCE");
    
    // Cache sensing start in seconds since reference epoch
    double t0 = sensingStart.secondsSinceEpoch(this->refEpoch());
    // Adjust for const azimuth shift
    t0 -= (azshift - 0.5 * (mode.numberAzimuthLooks() - 1)) / mode.prf();

    // Cache starting range
    double r0 = mode.startingRange();
    // Adjust for constant range shift
    r0 -= (rgshift - 0.5 * (mode.numberRangeLooks() - 1)) * mode.rangePixelSpacing();

    // Compute azimuth time extents
    double dtaz = mode.numberAzimuthLooks() / mode.prf();
    const double tend = t0 + ((mode.length() - 1) * dtaz);
    const double tmid = 0.5 * (t0 + tend);

    // Compute range extents
    const double dmrg = mode.numberRangeLooks() * mode.rangePixelSpacing();
    const double rngend = r0 + ((mode.width() - 1) * dmrg);

    // Print out extents
    _printExtents(info, t0, tend, dtaz, r0, rngend, dmrg, demWidth, demLength);

    // Interpolate orbit to middle of the scene as a test
    _checkOrbitInterpolation(tmid);

    // Compute number of lines per block
    computeLinesPerBlock();

    // Compute number of DEM blocks needed to process image
    size_t nBlocks = demLength / _linesPerBlock;
    if ((demLength % _linesPerBlock) != 0)
        nBlocks += 1;

    // Loop over blocks
    unsigned int totalconv = 0;
    for (size_t block = 0; block < nBlocks; ++block) {

        // Get block extents
        size_t lineStart, blockLength;
        lineStart = block * _linesPerBlock;
        if (block == (nBlocks - 1)) {
            blockLength = demLength - lineStart;
        } else {
            blockLength = _linesPerBlock;
        }
        size_t blockSize = blockLength * demWidth;

        // Diagnostics
        info << "Processing block: " << block << " " << pyre::journal::newline
             << "  - line start: " << lineStart << pyre::journal::newline
             << "  - line end  : " << lineStart + blockLength << pyre::journal::newline
             << "  - dopplers near mid far: "
             << doppler.eval(0, 0) << " "
             << doppler.eval(0, (mode.width() / 2) - 1) << " "
             << doppler.eval(0, mode.width() - 1) << " "
             << pyre::journal::endl;

        // Valarrays to hold input block from topo rasters
        std::valarray<double> x(blockSize), y(blockSize), hgt(blockSize);
        // Valarrays to hold block of geo2rdr results
        std::valarray<float> rgoff(blockSize), azoff(blockSize);

        // Read block of topo data
        topoRaster.getBlock(x, 0, lineStart, demWidth, blockLength, 1);
        topoRaster.getBlock(y, 0, lineStart, demWidth, blockLength, 2);
        topoRaster.getBlock(hgt, 0, lineStart, demWidth, blockLength,3);

        // Process block on GPU
        isce::cuda::geometry::runGPUGeo2rdr(
            ellipsoid, orbit, doppler, mode, x, y, hgt, azoff, rgoff, topoEPSG,
            lineStart, demWidth, t0, r0, this->threshold(), this->numiter(), totalconv
        );

        // Write block of data
        rgoffRaster.setBlock(rgoff, 0, lineStart, demWidth, blockLength);
        azoffRaster.setBlock(azoff, 0, lineStart, demWidth, blockLength);

    } // end for loop blocks in DEM image
            
    // Print out convergence statistics
    info << "Total convergence: " << totalconv << " out of "
         << (demWidth * demLength) << pyre::journal::endl;

}

// Print extents and image sizes
void isce::cuda::geometry::Geo2rdr::
_printExtents(pyre::journal::info_t & info, double t0, double tend, double dtaz,
              double r0, double rngend, double dmrg, size_t demWidth, size_t demLength) {
    info << pyre::journal::newline
         << "Starting acquisition time: " << t0 << pyre::journal::newline
         << "Stop acquisition time: " << tend << pyre::journal::newline
         << "Azimuth line spacing in seconds: " << dtaz << pyre::journal::newline
         << "Near range (m): " << r0 << pyre::journal::newline
         << "Far range (m): " << rngend << pyre::journal::newline
         << "Radar image length: " << this->mode().length() << pyre::journal::newline
         << "Radar image width: " << this->mode().width() << pyre::journal::newline
         << "Geocoded lines: " << demLength << pyre::journal::newline
         << "Geocoded samples: " << demWidth << pyre::journal::newline;
}

// Check we can interpolate orbit to middle of DEM
void isce::cuda::geometry::Geo2rdr::
_checkOrbitInterpolation(double aztime) {
    isce::core::cartesian_t satxyz, satvel;
    Orbit orbit = this->orbit();
    int stat = orbit.interpolate(aztime, satxyz, satvel, this->orbitMethod());
    if (stat != 0) {
        pyre::journal::error_t error("isce.cuda.core.Geo2rdr");
        error
            << pyre::journal::at(__HERE__)
            << "Error in Topo::topo - Error getting state vector for bounds computation."
            << pyre::journal::newline
            << " - requested time: " << aztime << pyre::journal::newline
            << " - bounds: " << orbit.UTCtime[0] << " -> " << orbit.UTCtime[orbit.nVectors-1]
            << pyre::journal::endl;
    }
}

// Compute number of lines per block dynamically from GPU memory
void isce::cuda::geometry::Geo2rdr::
computeLinesPerBlock() {

    // Compute GPU memory
    const size_t nGPUBytes = getDeviceMem();

    // 2 GB buffer for safeguard for large enough devices (> 6 GB)
    size_t gpuBuffer;
    if (nGPUBytes > 6e9) {
        gpuBuffer = 2e9;
    } else {
        // Else use 500 MB buffer
        gpuBuffer = 500e6;
    }

    // Compute pixels per Block (3 double input layers and 2 float output layers)
    size_t pixelsPerBlock = (nGPUBytes - gpuBuffer) /
                            (3 * sizeof(double) + 2 * sizeof(float));
    // Round down to nearest 10 million pixels
    pixelsPerBlock = (pixelsPerBlock / 10000000) * 10000000;

    // Compute number of lines per block
    _linesPerBlock = pixelsPerBlock / this->mode().width();
    // Round down to nearest 500 lines
    _linesPerBlock = (_linesPerBlock / 500) * 500;
}

// end of file
