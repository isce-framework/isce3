//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan V. Riel
// Copyright 2017-2018
//

#include <iostream>
#include <cmath>

// isce::core
#include "Constants.h"
#include "ResampSlc.h"
using isce::core::SINC_HALF;
using isce::core::SINC_LEN;
using isce::core::SINC_ONE;
using isce::core::SINC_SUB;

#define LINES_PER_TILE 1000

// Main resamp entry point
void isce::core::ResampSlc::resamp(
    const std::string & inputFilename,          // filename of input SLC
    const std::string & outputFilename,         // filename of output resampled SLC
    const std::string & rgOffsetFilename,       // filename of range offsets
    const std::string & azOffsetFilename,       // filename of azimuth offsets
    bool flatten, bool isComplex, size_t rowBuffer) {

    // Initialize journal channels
    pyre::journal::info_t infoChannel("isce.core.ResampSlc");
    pyre::journal::error_t errorChannel("isce.core.ResampSlc");

    // Check if data are not complex
    if (!isComplex) {
        errorChannel
            << pyre::journal::at(__HERE__)
            << "Real data interpolation not implemented yet."
            << pyre::journal::newline
            << pyre::journal::endl;
        return;
    }
        
    // Make input rasters
    Raster inputSlc(inputFilename, true);
    Raster rgOffsetRaster(rgOffsetFilename, true);
    Raster azOffsetRaster(azOffsetFilename, true);
    // Cache width of SLC image
    const size_t inLength = inputSlc.length();
    const size_t inWidth = inputSlc.width();
    // Cache output length and width from offset images
    const size_t outLength = rgOffsetRaster.length();
    const size_t outWidth = rgOffsetRaster.width();

    // Make output raster
    Raster outputSlc(outputFilename, outWidth, outLength, 1, GDT_CFloat32, "ISCE");

    // Save starting processing time
    const double procT0 = omp_get_wtime();

    // Announce myself to the world
    declare(inLength, inWidth, outLength, outWidth);

    // Initialize resampling methods
    _prepareMethods(SINC_METHOD);
   
    // Determine number of tiles needed to process image
    const size_t nTiles = _computeNumberOfTiles(LINES_PER_TILE);
    infoChannel << "Resampling using " << nTiles << " of " << LINES_PER_TILE << " lines"
        << pyre::journal::newline << pyre::journal::endl;

    // For each full tile of LINES_PER_TILE lines...
    size_t outputLine = 0;
    for (size_t tileCount = 0; tileCount < nTiles; tile++) {

        // Make a tile for representing input SLC data
        Tile_t tile;
        tile.width(inWidth);
        // Set its line index bounds (line number in output image)
        tile.rowStart = tileCount * LINES_PER_TILE;
        if (tileCount == (nTiles - 1)) {
            tile.rowEnd = outLength - tile.rowStart + 1;
        } else {
            tile.rowEnd = tile.rowStart + LINES_PER_TILE;
        }
       
        // Get corresponding image indices
        infoChannel << "Reading in image data for tile " << tileCount << pyre::journal::endl;
        _initializeTile(tile, azOffsetRaster, rowBuffer); 
    
        // Perform interpolation
        infoChannel << "Interpolating tile " << tileCount << pyre::journal::endl;
        _transformTile(tile, outputSlc, rgOffsetRaster, azOffsetRaster, outputLine);
    }
    infoChannel << "Elapsed time: " << (omp_get_wtime() - procT0) << " seconds"
        << pyre::journal::endl;
}

// Initialize tile bounds
void isce::core::ResampSlc::_initializeTile(Tile_t & tile, Raster & inputSlc,
    Raster & azOffsetRaster, size_t rowBuffer) {

    // Cache geometry values
    const size_t inLength = inputSlc.length();
    const size_t inWidth = inputSlc.width();
    const size_t outLength = azOffsetRaster.length();
    const size_t outWidth = azOffsetRaster.width();
    
    // Allocate array for storing residual azimuth
    std::valarray<double> residAz(outWidth, 0.0);

    // Compute minimum row index needed from input image
    tile.firstImageRow(outLength - 1);
    // Iterate over first rowBuffer lines of tile
    for (size_t i = tile.rowStart; i < (tile.rowStart + rowBuffer); ++i) {
        // Read in azimuth residual
        azOffsetRaster.getLine(&residAz[0], i, outWidth);
        // Now iterate over width of the tile
        for (int j = 0; j < outWidth; ++j) {
            // Compute total azimuth offset of current pixel
            double azOff = _azOffsetsPoly.eval(i+1, j+1) + residAz[j];
            // Calculate corresponding minimum line index of input image
            size_t imageLine = size_t(i + azOff) - SINC_HALF;
            // Update minimum row index
            tile.firstImageRow(std::min(tile.firstImageRow(), imageLine));
        }
    }
    // Final update
    tile.firstImageRow(std::max(tile.firstImageRow(), 0));

    // Compute maximum row index needed from input image
    tile.lastImageRow(0);
    // Iterate over last rowBuffer lines of tile
    for (size_t i = (tile.rowStart - rowBuffer); i < tile.rowEnd; ++i) {
        // Read in azimuth residual
        azOffsetRaster.getLine(residAz, i, outWidth);
        // Now iterate over width of the tile
        for (size_t j = 0; j < outWidth; j++) {
            // Compute total azimuth offset of current pixel
            double azOff = _azOffsetsPoly.eval(i+1, j+1) + residAz[j];
            // Calculate corresponding minimum line index of input image
            size_t imageLine = size_t(i + azOff) + SINC_HALF;
            // Update maximum row index
            tile.lastImageRow(std::max(tile.lastImageRow(), imageLine));
        }
    }
    // Final udpate
    tile.lastImageRow(std::min(tile.lastImageRow(), inLength - 1));

    // Tile will allocate memory for itself
    tile.allocate();

    // Read in tile.length() lines of data from the input image to the image block
    for (size_t i = 0; i < tile.length(); i++) {
        // Read line of data
        tile.setLineData(inputSlc, tile.firstImageRow() + i);
        // Remove the carrier phases in parallel
        #pragma omp parallel for
        for (size_t j = 0; j < inWidth; j++) {
            // Evaluate the pixel's carrier phase
            double phase = modulo_f(
                  _rgCarrier.eval(tile.firstImageRow() + i + 1, j + 1) 
                + _azCarrier.eval(tile.firstImageRow() + i + 1, j + 1), 2.0*M_PI);
            // Remove the carrier
            std::complex<float> cpxPhase(std::cos(phase), -std::sin(phase));
            tile[IDX1D(i,j,inWidth)] *= cpxPhase;
        }
    }
}

// Interpolate tile to perform transformation
void isce::core::ResampSlc::_transformTile(Tile_t & tile, Raster & outputSlc,
    Raster & rgOffsetRaster, Raster & azOffsetRaster, size_t & outputLine) {

    // Cache geometry values
    const size_t inLength = inputSlc.length();
    const size_t inWidth = inputSlc.width();
    const size_t outLength = azOffsetRaster.length();
    const size_t outWidth = azOffsetRaster.width();

    // Allocate valarrays for work
    std::valarray<double> residAz(0.0, outWidth), residRg(0.0, outWidth);
    std::valarray<std::complex<float>> chip(SINC_ONE * SINC_ONE);
    std::valarray<std::complex<float>> imgOut(outWidth);
    
    // Loop over lines to perform interpolation
    for (size_t i = tile.rowStart; i < tile.rowEnd; ++i) {
        // Get lines for residual offsets
        rgOffsetRaster.getLine(&residRg[0], i, outWidth);
        azOffsetRaster.getLine(&residAz[0], i, outWidth);
        // Loop over width
        #pragma omp parallel for firstPrivate(chip)
        for (size_t j = 0; j < outWidth; ++j) {
           
            // Evaluate offset polynomials 
            const double azOff = _azOffsetsPoly.eval(i+1, j+1) + residAz[j];
            const double rgOff = _rgOffsetsPoly.eval(i+1, j+1) + residRg[j];
            // Break into fractional and integer parts
            size_t k, kk;
            const double fracAz = std::modf(i + azOff, &k);
            const double fracRg = std::modf(j + rgOff, &kk);
            // Check bounds
            if ((k < SINC_HALF) || (k >= (inLength - SINC_HALF))) continue;
            if ((kk < SINC_HALF) || (kk >= (inWidth  -SINC_HALF))) continue;

            // Evaluate Doppler polynomial
            const double dop = _dopplerPoly.eval(i+1, j+1);

            // Data chip without the carrier phases
            for (size_t ii = 0; ii < SINC_ONE; ++ii) {
                // Subtracting off firstImageRow removes the offset from the first row
                // in the master image to the first row actually contained in input tile
                const size_t chipRow = k - tile.firstImageRow() + ii - SINC_HALF;
                const double phase = dop * (ii - 4.0);
                const std::complex<float> cval(std::cos(phase), -std::sin(phase));
                // Set the data values after removing doppler in azimuth
                for (size_t jj = 0; jj < SINC_ONE; ++jj) {
                    const size_t chipCol = kk + jj - SINC_HALF;
                    chip[IDX1D(ii,jj,SINC_ONE)] = tile[IDX1D(chipRow,chipCol,inWidth)] * cval;
                }
            }

            // Doppler to be added back. Simultaneously evaluate carrier that needs to
            // be added back after interpolation
            double phase = (dop * fracAz) + _rgCarrier.eval(i + azOff, j + rgOff) 
                + _azCarrier.eval(i + azOff, j + rgOff);

            // Flatten the carrier phase if requested
            if (flatten) {
                phase += ((4. * (M_PI / meta.radarWavelength)) * 
                    ((meta.rangeFirstSample - refMeta.rangeFirstSample) 
                    + (j * (meta.slantRangePixelSpacing - refMeta.slantRangePixelSpacing)) 
                    + (rgOff * meta.slantRangePixelSpacing))) + ((4.0 * M_PI 
                    * (refMeta.rangeFirstSample + (j * refMeta.slantRangePixelSpacing))) 
                    * ((1.0 / refMeta.radarWavelength) - (1.0 / meta.radarWavelength)));
            }
            // Modulate by 2*PI
            phase = modulo_f(phase, 2.0*M_PI);
           
            // Interpolate 
            cval = _interpolateComplex(chip, (SINC_HALF + 1), (SINC_HALF + 1),
                fracAz, fracRg, SINC_ONE, SINC_ONE, SINC_METHOD);

            // Add doppler to interpolated value and save
            imgOut[j] = cval * std::complex<float>(std::cos(phase), std::sin(phase));
        } // end for over width

        // Write out line of SLC data and increment output line index
        outputSlc.setLine(&imgOut[0], outputLine, outWidth);
        outputLine += 1;

    } // end for over length
}

// end of file
