//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan V. Riel, Liang Yu
// Copyright 2017-2018
//

#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>

// isce::core
#include "isce/core/Constants.h"

#include "ResampSlc.h"
#include "gpuResampSlc.h"

using isce::io::Raster;

// Main product-based resamp entry point
void isce::cuda::image::ResampSlc::
resamp(const std::string & outputFilename,
       const std::string & polarization,
       const std::string & rgOffsetFilename,
       const std::string & azOffsetFilename,
       bool flatten, bool isComplex, int rowBuffer,
       int chipSize) {

    // Form the GDAL-compatible path for the HDF5 dataset
    const std::string dataPath = _mode.dataPath(polarization);
    const std::string h5path = "HDF5:\"" + _filename + "\":/" + dataPath;

    // Call alternative resmap entry point using filenames
    resamp(h5path, outputFilename, rgOffsetFilename, azOffsetFilename, 1,
           flatten, isComplex, rowBuffer, chipSize);
}

// Alternative generic resamp entry point: use filenames to internally create rasters
void isce::cuda::image::ResampSlc::
resamp(const std::string & inputFilename,          // filename of input SLC
       const std::string & outputFilename,         // filename of output resampled SLC
       const std::string & rgOffsetFilename,       // filename of range offsets
       const std::string & azOffsetFilename,       // filename of azimuth offsets
       int inputBand, bool flatten, bool isComplex, int rowBuffer,
       int chipSize) {

    // Make input rasters
    Raster inputSlc(inputFilename, GA_ReadOnly);
    Raster rgOffsetRaster(rgOffsetFilename, GA_ReadOnly);
    Raster azOffsetRaster(azOffsetFilename, GA_ReadOnly);

    // Make output raster; geometry defined by offset rasters
    const int outLength = rgOffsetRaster.length();
    const int outWidth = rgOffsetRaster.width();
    Raster outputSlc(outputFilename, outWidth, outLength, 1, GDT_CFloat32, "ISCE");

    // Call generic resamp
    resamp(inputSlc, outputSlc, rgOffsetRaster, azOffsetRaster, inputBand, flatten,
           isComplex, rowBuffer, chipSize);
}

// Generic resamp entry point from externally created rasters
void isce::cuda::image::ResampSlc::
resamp(isce::io::Raster & inputSlc, isce::io::Raster & outputSlc,
       isce::io::Raster & rgOffsetRaster, isce::io::Raster & azOffsetRaster,
       int inputBand, bool flatten, bool isComplex, int rowBuffer,
       int chipSize) {

    // Check if data are not complex
    if (!isComplex) {
        std::cout << "Real data interpolation not implemented yet.\n";
        return;
    }
    // Set the band number for input SLC
    _inputBand = inputBand;
    // Cache width of SLC image
    const int inLength = inputSlc.length();
    const int inWidth = inputSlc.width();
    // Cache output length and width from offset images
    const int outLength = rgOffsetRaster.length();
    const int outWidth = rgOffsetRaster.width();

    // initialize interpolator
    isce::cuda::core::gpuSinc2dInterpolator<gpuComplex<float>> interp(chipSize-1, isce::core::SINC_SUB);

    // Determine number of tiles needed to process image
    const int nTiles = _computeNumberOfTiles(outLength, _linesPerTile);
    std::cout << 
        "GPU resampling using " << nTiles << " tiles of " << _linesPerTile 
        << " lines per tile\n";
    // Start timer
    auto timerStart = std::chrono::steady_clock::now();

    // For each full tile of _linesPerTile lines...
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

        // Initialize offsets tiles
        isce::image::Tile<float> azOffTile, rgOffTile;
        _initializeOffsetTiles(tile, azOffsetRaster, rgOffsetRaster,
                               azOffTile, rgOffTile, outWidth);

        // Get corresponding image indices
        std::cout << "Reading in image data for tile " << tileCount << std::endl;
        _initializeTile(tile, inputSlc, azOffTile, outLength, rowBuffer, chipSize/2); 
    
        // Perform interpolation
        std::cout << "Interpolating tile " << tileCount << std::endl;
        gpuTransformTile(tile, outputSlc, rgOffTile, azOffTile, _rgCarrier, _azCarrier, 
                _dopplerLUT, _mode, _refMode, _haveRefMode, interp, inWidth, inLength, flatten, chipSize);
    }

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
        timerEnd - timerStart).count();
    std::cout << "Elapsed processing time: " << elapsed << " sec" << "\n";
}

// end of file
