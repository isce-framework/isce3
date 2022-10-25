// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#include "Geo2rdr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <valarray>

#include <isce3/core/Constants.h>

#include "geometry.h"

// pull in some isce3::core namespaces
using isce3::io::Raster;
using isce3::core::LUT1d;
using isce3::core::Vec3;

// Run geo2rdr with no offsets; internal creation of offset rasters
void isce3::geometry::Geo2rdr::
geo2rdr(isce3::io::Raster & topoRaster,
        const std::string & outdir,
        double azshift, double rgshift)
{
    // Cache the size of the DEM images
    const size_t demWidth = topoRaster.width();
    const size_t demLength = topoRaster.length();

    // Create output rasters
    Raster rgoffRaster = Raster(outdir + "/range.off", demWidth, demLength, 1,
        GDT_Float64, "ISCE");
    Raster azoffRaster = Raster(outdir + "/azimuth.off", demWidth, demLength, 1,
        GDT_Float64, "ISCE");

    // Call main geo2rdr with offsets set to zero
    geo2rdr(topoRaster, rgoffRaster, azoffRaster, azshift, rgshift);
}

// Run geo2rdr with externally created offset rasters
void isce3::geometry::Geo2rdr::
geo2rdr(isce3::io::Raster & topoRaster,
        isce3::io::Raster & rgoffRaster,
        isce3::io::Raster & azoffRaster,
        double azshift, double rgshift)
{
    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.geometry.Geo2rdr");
    pyre::journal::info_t info("isce.geometry.Geo2rdr");

    // Cache the size of the DEM images
    const size_t demWidth = topoRaster.width();
    const size_t demLength = topoRaster.length();

    // Initialize projection for topo results
    _projTopo = isce3::core::createProj(topoRaster.getEPSG());

    // Cache sensing start
    double t0 = _radarGrid.sensingStart();
    // Adjust for const azimuth shift
    t0 -= azshift / _radarGrid.prf();

    // Cache starting range
    double r0 = _radarGrid.startingRange();
    // Adjust for constant range shift
    r0 -= rgshift * _radarGrid.rangePixelSpacing();

    // Compute azimuth time extents
    double dtaz = 1.0 / _radarGrid.prf();
    const double tend = t0 + ((_radarGrid.length() - 1) * dtaz);
    const double tmid = 0.5 * (t0 + tend);

    // Compute range extents
    const double dmrg = _radarGrid.rangePixelSpacing();
    const double rngend = r0 + ((_radarGrid.width() - 1) * dmrg);

    // Print out extents
    _printExtents(info, t0, tend, dtaz, r0, rngend, dmrg, demWidth, demLength);

    // Interpolate orbit to middle of the scene as a test
    _checkOrbitInterpolation(tmid);

    // Adjust block size if DEM has too few lines
    _linesPerBlock = std::min(demLength, _linesPerBlock);

    // Compute number of DEM blocks needed to process image
    size_t nBlocks = demLength / _linesPerBlock;
    if ((demLength % _linesPerBlock) != 0)
        nBlocks += 1;

    // Loop over blocks
    size_t converged = 0;
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
        const double tblock = _radarGrid.sensingTime(lineStart);
        info << "Processing block: " << block << " " << pyre::journal::newline
             << "  - line start: " << lineStart << pyre::journal::newline
             << "  - line end  : " << lineStart + blockLength << pyre::journal::newline
             << "  - dopplers near mid far: "
             << _doppler.eval(tblock, r0) << " "
             << _doppler.eval(tblock, 0.5*(r0 + rngend)) << " "
             << _doppler.eval(tblock, rngend) << " "
             << pyre::journal::endl;

        // Valarrays to hold input block from topo rasters
        std::valarray<double> x(blockSize), y(blockSize), hgt(blockSize);
        // Valarrays to hold block of geo2rdr results
        std::valarray<double> rgoff(blockSize), azoff(blockSize);

        // Read block of topo data
        topoRaster.getBlock(x, 0, lineStart, demWidth, blockLength, 1);
        topoRaster.getBlock(y, 0, lineStart, demWidth, blockLength, 2);
        topoRaster.getBlock(hgt, 0, lineStart, demWidth, blockLength,3);

        // Loop over DEM lines in block
        for (size_t blockLine = 0; blockLine < blockLength; ++blockLine) {

            // Global line index
            const size_t line = lineStart + blockLine;

            // Loop over DEM pixels
            #pragma omp parallel for reduction(+:converged)
            for (size_t pixel = 0; pixel < demWidth; ++pixel) {

                // Convert topo XYZ to LLH
                const size_t index = blockLine * demWidth + pixel;
                Vec3 xyz{x[index], y[index], hgt[index]};
                Vec3 llh = _projTopo->inverse(xyz);

                // Perform geo->rdr iterations
                double aztime, slantRange;
                int geostat = isce3::geometry::geo2rdr(
                    llh, _ellipsoid, _orbit, _doppler,  aztime, slantRange,
                    _radarGrid.wavelength(), _radarGrid.lookSide(),
                    _threshold, _numiter, 1.0e-8
                );

                // Check if solution is out of bounds
                bool isOutside = false;
                if ((aztime < t0) || (aztime > tend))
                    isOutside = true;
                if ((slantRange < r0) || (slantRange > rngend))
                    isOutside = true;

                // Save result if valid
                if (!isOutside) {
                    rgoff[index] = ((slantRange - r0) / dmrg) - static_cast<double>(pixel);
                    azoff[index] = ((aztime - t0) / dtaz) - static_cast<double>(line);
                    converged += geostat;
                } else {
                    rgoff[index] = NULL_VALUE;
                    azoff[index] = NULL_VALUE;
                }
            } // end OMP for loop pixels in block
        } // end for loop lines in block

        // Write block of data
        rgoffRaster.setBlock(rgoff, 0, lineStart, demWidth, blockLength);
        azoffRaster.setBlock(azoff, 0, lineStart, demWidth, blockLength);

    } // end for loop blocks in DEM image

    // Print out convergence statistics
    info << "Total convergence: " << converged << " out of "
         << (demWidth * demLength) << pyre::journal::endl;
}

// Print extents and image sizes
void isce3::geometry::Geo2rdr::
_printExtents(pyre::journal::info_t & info, double t0, double tend, double dtaz,
              double r0, double rngend, double dmrg, size_t demWidth, size_t demLength)
{
    info << pyre::journal::newline
         << "Starting acquisition time: " << t0 << pyre::journal::newline
         << "Stop acquisition time: " << tend << pyre::journal::newline
         << "Azimuth line spacing in seconds: " << dtaz << pyre::journal::newline
         << "Slant range spacing in meters: " << dmrg << pyre::journal::newline
         << "Near range (m): " << r0 << pyre::journal::newline
         << "Far range (m): " << rngend << pyre::journal::newline
         << "Radar image length: " << _radarGrid.length() << pyre::journal::newline
         << "Radar image width: " << _radarGrid.width() << pyre::journal::newline
         << "Geocoded lines: " << demLength << pyre::journal::newline
         << "Geocoded samples: " << demWidth << pyre::journal::newline;
}

// Check we can interpolate orbit to middle of DEM
void isce3::geometry::Geo2rdr::
_checkOrbitInterpolation(double aztime)
{
    Vec3 pos, vel;
    _orbit.interpolate(&pos, &vel, aztime);
}

// end of file
