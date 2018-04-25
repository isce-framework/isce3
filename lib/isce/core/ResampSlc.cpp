//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan V. Riel
// Copyright 2017-2018
//

#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>

// pyre
#include <portinfo>
#include <pyre/journal.h>

// isce::core
#include "Constants.h"
#include "Interpolator.h"
#include "ResampSlc.h"
using isce::core::SINC_HALF;
using isce::core::SINC_LEN;
using isce::core::SINC_ONE;
using isce::core::SINC_SUB;

// Main resamp entry point
void isce::core::ResampSlc::
resamp(const std::string & inputFilename,          // filename of input SLC
       const std::string & outputFilename,         // filename of output resampled SLC
       const std::string & rgOffsetFilename,       // filename of range offsets
       const std::string & azOffsetFilename,       // filename of azimuth offsets
       bool flatten, bool isComplex, int rowBuffer) {

    // Initialize journal channel for info
    pyre::journal::info_t infoChannel("isce.core.ResampSlc");

    // Check if data are not complex
    if (!isComplex) {
        pyre::journal::error_t errorChannel("isce.core.ResampSlc");
        errorChannel
            << pyre::journal::at(__HERE__)
            << "Real data interpolation not implemented yet."
            << pyre::journal::newline
            << pyre::journal::endl;
        return;
    }
        
    // Make input rasters
    Raster inputSlc(inputFilename, GA_ReadOnly);
    Raster rgOffsetRaster(rgOffsetFilename, GA_ReadOnly);
    Raster azOffsetRaster(azOffsetFilename, GA_ReadOnly);
    // Cache width of SLC image
    const int inLength = inputSlc.length();
    const int inWidth = inputSlc.width();
    // Cache output length and width from offset images
    const int outLength = rgOffsetRaster.length();
    const int outWidth = rgOffsetRaster.width();

    // Make output raster
    Raster outputSlc(outputFilename, outWidth, outLength, 1, GDT_CFloat32, "ISCE");

    // Announce myself to the world
    declare(inLength, inWidth, outLength, outWidth);

    // Initialize resampling methods
    _prepareInterpMethods(SINC_METHOD);
   
    // Determine number of tiles needed to process image
    const int nTiles = _computeNumberOfTiles(outLength, _linesPerTile);
    infoChannel << 
        "Resampling using " << nTiles << " tiles of " << _linesPerTile 
        << " lines per tile"
        << pyre::journal::newline << pyre::journal::endl;

    // Start timer
    auto timerStart = std::chrono::steady_clock::now();

    // For each full tile of _linesPerTile lines...
    int outputLine = 0;
    for (int tileCount = 0; tileCount < nTiles; tileCount++) {

        // Make a tile for representing input SLC data
        Tile_t tile;
        tile.width(inWidth);
        // Set its line index bounds (line number in output image)
        tile.rowStart(tileCount * _linesPerTile);
        if (tileCount == (nTiles - 1)) {
            tile.rowEnd(outLength);
        } else {
            tile.rowEnd(tile.rowStart() + _linesPerTile);
        }
    
        // Get corresponding image indices
        infoChannel << "Reading in image data for tile " << tileCount << pyre::journal::newline;
        _initializeTile(tile, inputSlc, azOffsetRaster, rowBuffer); 
        // Send some diagnostics to the journal
        tile.declare(infoChannel);
    
        // Perform interpolation
        infoChannel << "Interpolating tile " << tileCount << pyre::journal::endl;
        _transformTile(tile, outputSlc, rgOffsetRaster, azOffsetRaster, inLength,
            flatten, outputLine);
    }

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
        timerEnd - timerStart).count();
    infoChannel << "Elapsed processing time: " << elapsed << " sec"
                << pyre::journal::endl;
}

// Initialize tile bounds
void isce::core::ResampSlc::
_initializeTile(Tile_t & tile, Raster & inputSlc, Raster & azOffsetRaster, int rowBuffer) {

    // Cache geometry values
    const int inLength = inputSlc.length();
    const int inWidth = inputSlc.width();
    const int outLength = azOffsetRaster.length();
    const int outWidth = azOffsetRaster.width();
    
    // Allocate array for storing residual azimuth
    std::valarray<double> residAz(outWidth);

    // Compute minimum row index needed from input image
    tile.firstImageRow(outLength - 1);
    // Iterate over first rowBuffer lines of tile
    for (int i = tile.rowStart();
        i < std::min(tile.rowEnd(), tile.rowStart() + rowBuffer); ++i) {
        // Read in azimuth residual
        azOffsetRaster.getLine(residAz, i);
        // Now iterate over width of the tile
        for (int j = 0; j < outWidth; ++j) {
            // Compute total azimuth offset of current pixel
            const double azOff = residAz[j];
            // Calculate corresponding minimum line index of input image
            const int imageLine = static_cast<int>(i + azOff) - SINC_HALF;
            // Update minimum row index
            tile.firstImageRow(std::min(tile.firstImageRow(), imageLine));
        }
    }
    // Final update
    tile.firstImageRow(std::max(tile.firstImageRow(), 0));
    
    // Compute maximum row index needed from input image
    tile.lastImageRow(0);
    // Iterate over last rowBuffer lines of tile
    for (int i = std::max(tile.rowStart() - rowBuffer, 0);
        i < std::min(outLength, tile.rowEnd()); ++i) {
        // Read in azimuth residual
        azOffsetRaster.getLine(residAz, i);
        // Now iterate over width of the tile
        for (int j = 0; j < outWidth; j++) {
            // Compute total azimuth offset of current pixel
            const double azOff = residAz[j];
            // Calculate corresponding minimum line index of input image
            const int imageLine = static_cast<int>(i + azOff) + SINC_HALF;
            // Update maximum row index
            tile.lastImageRow(std::max(tile.lastImageRow(), imageLine));
        }
    }
    // Final udpate
    tile.lastImageRow(std::min(tile.lastImageRow(), inLength - 1));

    // Tile will allocate memory for itself
    tile.allocate();

    // Read in tile.length() lines of data from the input image to the image block
    for (int i = 0; i < tile.length(); i++) {
        // Read line of data into tile
        inputSlc.getLine(&tile[i*tile.width()], tile.firstImageRow() + i, tile.width());
        // Remove the carrier phases in parallel
        //#pragma omp parallel for
        for (int j = 0; j < inWidth; j++) {
            // Evaluate the pixel's carrier phase
            const double phase = modulo_f(
                  _rgCarrier.eval(tile.firstImageRow() + i, j) 
                + _azCarrier.eval(tile.firstImageRow() + i, j), 2.0*M_PI);
            // Remove the carrier
            std::complex<float> cpxPhase(std::cos(phase), -std::sin(phase));
            tile(i,j) *= cpxPhase;
        }
    }
}

// Interpolate tile to perform transformation
void isce::core::ResampSlc::
_transformTile(Tile_t & tile, Raster & outputSlc, Raster & rgOffsetRaster,
               Raster & azOffsetRaster, int inLength, bool flatten, int & outputLine) {

    // Cache geometry values
    const int inWidth = tile.width();
    const int outWidth = azOffsetRaster.width();

    // Allocate valarrays for work
    std::valarray<float> residAz(outWidth), residRg(outWidth);
    std::valarray<std::complex<float>> imgOut(outWidth);
    Matrix<std::complex<float>> chip(SINC_ONE, SINC_ONE);
    
    // Loop over lines to perform interpolation
    for (int i = tile.rowStart(); i < tile.rowEnd(); ++i) {

        // Initialize output line to zeros
        imgOut = std::complex<float>(0.0, 0.0);

        // Get lines for residual offsets
        rgOffsetRaster.getLine(residRg, i);
        azOffsetRaster.getLine(residAz, i);

        // Loop over width
        #pragma omp parallel for firstprivate(chip)
        for (int j = 0; j < outWidth; ++j) {
           
            // Unpack offsets
            const float azOff = residAz[j];
            const float rgOff = residRg[j];

            // Break into fractional and integer parts
            const int intAz = static_cast<int>(i + azOff);
            const int intRg = static_cast<int>(j + rgOff);
            const double fracAz = i + azOff - intAz;
            const double fracRg = j + rgOff - intRg;
           
            // Check bounds
            if ((intAz < SINC_HALF) || (intAz >= (inLength - SINC_HALF)))
                continue;
            if ((intRg < SINC_HALF) || (intRg >= (inWidth - SINC_HALF)))
                continue;

            // Evaluate Doppler polynomial
            const double dop = _dopplerPoly.eval(0, j) * 2*M_PI / _meta.prf;

            // Doppler to be added back. Simultaneously evaluate carrier that needs to
            // be added back after interpolation
            double phase = (dop * fracAz) 
                + _rgCarrier.eval(i + azOff, j + rgOff) 
                + _azCarrier.eval(i + azOff, j + rgOff);

            // Flatten the carrier phase if requested
            if (flatten) {
                phase += ((4. * (M_PI / _meta.radarWavelength)) * 
                    ((_meta.rangeFirstSample - _refMeta.rangeFirstSample) 
                    + (j * (_meta.slantRangePixelSpacing - _refMeta.slantRangePixelSpacing)) 
                    + (rgOff * _meta.slantRangePixelSpacing))) + ((4.0 * M_PI 
                    * (_refMeta.rangeFirstSample + (j * _refMeta.slantRangePixelSpacing))) 
                    * ((1.0 / _refMeta.radarWavelength) - (1.0 / _meta.radarWavelength)));
            }
            // Modulate by 2*PI
            phase = modulo_f(phase, 2.0*M_PI);
            
            // Read data chip without the carrier phases
            for (int ii = 0; ii < SINC_ONE; ++ii) {
                // Row to read from
                const int chipRow = intAz - tile.firstImageRow() + ii - SINC_HALF;
                // Carrier phase
                const double phase = dop * (ii - 4.0);
                const std::complex<float> cval(std::cos(phase), -std::sin(phase));
                // Set the data values after removing doppler in azimuth
                for (int jj = 0; jj < SINC_ONE; ++jj) {
                    // Column to read from
                    const int chipCol = intRg + jj - SINC_HALF;
                    chip(ii,jj) = tile(chipRow,chipCol) * cval;
                }
            }

            // Interpolate fractional component
            const std::complex<float> cval = _interpolateComplex(
                chip, (SINC_HALF + 1), (SINC_HALF + 1), fracAz, fracRg, SINC_ONE, SINC_ONE
            );

            // Add doppler to interpolated value and save
            imgOut[j] = cval * std::complex<float>(std::cos(phase), std::sin(phase));

        } // end for over width

        // Write out line of SLC data and increment output line index
        outputSlc.setLine(imgOut, outputLine);
        outputLine += 1;

    } // end for over length
}

// end of file
