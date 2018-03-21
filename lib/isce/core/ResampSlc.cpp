//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan V. Riel
// Copyright 2017-2018
//

#include <iostream>
#include <cmath>
#include "ResampSlc.h"

// Main resamp entry point
void isce::core::ResampSlc::resamp(bool flatten, bool isComplex, size_t rowBuffer) {

    // Allocate memory for work arrays
    vector<vector<complex<float>>> chip(SINC_ONE, vector<complex<float> >(SINC_ONE));
    vector<complex<float>> imgIn(0);
    vector<complex<float>> imgOut(_outWidth, std::complex<float>(0.0, 0.0));
    
    DataAccessor *slcInAccObj = (DataAccessor*)slcInAccessor;
    DataAccessor *slcOutAccObj = (DataAccessor*)slcOutAccessor;
    DataAccessor *residRgAccObj, *residAzAccObj;
    if (residRgAccessor != 0) {
        residRgAccObj = (DataAccessor*) residRgAccessor;
    } else {
        residRgAccObj = NULL;
    }

    if (residAzAccessor != 0) {
        residAzAccObj = (DataAccessor*) residAzAccessor;
    } else {
        residAzAccObj = NULL;
    }

    // Moving this here so we don't waste any time
    if (!isComplex) {
        printf("Real data interpolation not implemented yet.\n");
        return;
    }

    // Save starting processing time
    double procT0 = omp_get_wtime();

    // Announce myself to the world
    declare();

    // Initialize resampling methods
    ResampMethods rMethods;
    rMethods.prepareMethods(SINC_METHOD);
   
    // Determine number of tiles needed to process image
    size_t nTiles = _computeNumberOfTiles(LINES_PER_TILE);
    std::cout << "Resampling using " << nTiles << " of " << LINES_PER_TILE << " lines\n\n";

    // For each full tile of LINES_PER_TILE lines...
    for (size_t tileCount = 0; tileCount < nTiles; tile++) {

        std::cout << "Reading in image data for tile " << tileCount << std::endl;

        // Make a new tile
        isce::core::Tile tile;
        // Set its line index bounds (line number in output image)
        tile.rowStart = tileCount * LINES_PER_TILE;
        if (tileCount == (nTiles - 1)) {
            tile.rowEnd = _outLength - tile.rowStart + 1;
        } else {
            tile.rowEnd = tile.rowStart + LINES_PER_TILE;
        }
       
        // Get corresponding image indices
        size_t firstImageRow, lastImageRow;
        _initializeTile(tile, rowBuffer, firstImageRow, lastImageRow); 
        // Number of rows in imgIn
        size_t nRowsInBlock = lastImageRow - firstImageRow + 1;

        // Resize the image tile to the necessary number of lines if necessary
        if (imgIn.size() < (nRowsInBlock * _inWidth)) {
            imgIn.resize(nRowsInBlock * _inWidth);
        }

        // Perform interpolation
        std::cout << "Interpolating tile " << tileCount << std::endl;
        _transformTile(tile, imgIn, imgOut, firstImageRow, nRowsInBlock);
        
    } // end for over tiles
    printf("Elapsed time: %f\n", (omp_get_wtime() - procT0));
}

// Initialize tile bounds
void isce::core::ResampSlc::_initializeTile(Tile<std::complex<float>> & tile,
    size_t rowBuffer, size_t & firstImageRow, size_t & lastImageRow) {

    std::vector<double> residAz(_outWidth, 0.0);

    // Compute minimum row index needed from input image
    firstImageRow = outLength - 1;
    // Iterate over first rowBuffer lines of tile
    for (size_t i = tile.rowStart; i < (tile.rowStart + rowBuffer); ++i) {
        // Read in azimuth residual if it exists
        if (_residAzAccessor != 0) _residAzAccObj->getLine((char *) &residAz[0], i);
        // Now iterate over width of the tile
        for (int j = 0; j < outWidth; ++j) {
            // Compute total azimuth offset of current pixel
            double azOff = _azOffsetsPoly->eval(i+1, j+1) + residAz[j];
            // Calculate corresponding minimum line index of input image
            size_t imageLine = size_t(i + azOff) - SINC_HALF;
            // Update minimum row index
            firstImageRow = std::min(firstImageRow, imageLine);
        }
    }
    // Final update
    firstImageRow = std::max(firstImageRow, 0);

    // Compute maximum row index needed from input image
    lastImageRow = 0;
    // Iterate over last rowBuffer lines of tile
    for (size_t i = (tile.rowStart - rowBuffer); i < tile.rowEnd; ++i) {
        // Read in azimuth residual if it exists
        if (_residAzAccessor != 0) _residAzAccObj->getLine((char *) &residAz[0], i);
        // Now iterate over width of the tile
        for (size_t j = 0; j < outWidth; j++) {
            // Compute total azimuth offset of current pixel
            double azOff = _azOffsetsPoly->eval(i+1, j+1) + residAz[j];
            // Calculate corresponding minimum line index of input image
            size_t imageLine = size_t(i + azOff) + SINC_HALF;
            // Update maximum row index
            lastImageRow = std::max(lastImageRow, imageLine);
        }
    }
    // Final udpate
    lastImageRow = std::min(lastImageRow, inLength - 1);
}

// Interpolate tile to perform transformation
void isce::core::ResampSlc::_transformTile(Tile<std::complex<float>> & tile,
    std::vector<std::complex<float>> & imgIn, std::vector<std::complex<float>> & imgOut,
    size_t firstImageRow, size_t nRowsInBlock) {

    std::vector<double> residAz(_outWidth, 0.0), residRg(_outWidth, 0.0);

    // Read in nRowsInBlock lines of data from the input image to the image block
    for (size_t i = 0; i < nRowsInBlock; i++) {
        // Read and set line of data
        slcInAccObj->getLine((char *) &imgIn[IDX1D(i,0,inWidth)], firstImageRow + i);
        // Remove the carrier phases in parallel
        #pragma omp parallel for
        for (size_t j = 0; j < _inWidth; j++) {
            // Evaluate the pixel's carrier phase
            double phase = modulo_f(_rgCarrier->eval(firstImageRow + i + 1, j + 1) 
                + _azCarrier->eval(firstImageRow + i + 1, j + 1), 2.0*M_PI);
            // Remove the carrier
            std::complex<float> cpxPhase(std::cos(phase), -std::sin(phase));
            imgIn[IDX1D(i,j,inWidth)] *= cpxPhase;
        }
    }
    
    // Loop over lines to perform interpolation
    for (size_t i = tile.rowStart; i < tile.rowEnd; ++i) {
        // Get next lines for residual offsets
        if (residAzAccessor != 0) _residAzAccObj->getLineSequential((char *) &residAz[0]);
        if (residRgAccessor != 0) _residRgAccObj->getLineSequential((char *) &residRg[0]);
        // Loop over width
        #pragma omp parallel for firstPrivate(chip)
        for (size_t j = 0; j < _outWidth; ++j) {
           
            // Evaluate offset polynomials 
            const double azOff = _azOffsetsPoly->eval(i+1, j+1) + residAz[j];
            const double rgOff = _rgOffsetsPoly->eval(i+1, j+1) + residRg[j];
            // Break into fractional and integer parts
            size_t k, kk;
            const double fracAz = std::modf(i + azOff, &k);
            const double fracRg = std::modf(j + rgOff, &kk);
            // Check bounds
            if ((k < SINC_HALF) || (k >= (_inLength - SINC_HALF))) continue;
            if ((kk < SINC_HALF) || (kk >= (_inWidth  -SINC_HALF))) continue;

            // Evaluate Doppler polynomial
            const double dop = _dopplerPoly->eval(i+1, j+1);

            // Data chip without the carrier phases
            for (size_t ii = 0; ii < SINC_ONE; ++ii) {
                // Subtracting off firstImageRow removes the offset from the first row
                // in the master image to the first row actually contained in imgIn
                const size_t chipRow = k - firstImageRow + ii - SINC_HALF;
                const double phase = dop * (ii - 4.0);
                const std::complex<float> cval(std::cos(phase), -std::sin(phase));
                // Set the data values after removing doppler in azimuth
                for (size_t jj = 0; jj < SINC_ONE; ++jj) {
                    const size_t chipCol = kk + jj - SINC_HALF;
                    chip[ii][jj] = imgIn[IDX1D(chipRow,chipCol,inWidth)] * cval;
                }
            }

            // Doppler to be added back. Simultaneously evaluate carrier that needs to
            // be added back after interpolation
            double phase = (dop * fracAz) + _rgCarrier->eval(i + azOff, j + rgOff) 
                + _azCarrier->eval(i + azOff, j + rgOff);

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
            cval = rMethods.interpolate_cx(chip, (SINC_HALF + 1), (SINC_HALF + 1),
                fracAz, fracRg, SINC_ONE, SINC_ONE, SINC_METHOD);

            // Add doppler to interpolated value and save
            imgOut[j] = cval * std::complex<float>(std::cos(phase), std::sin(phase));
        } // end for over width
        slcOutAccObj->setLineSequential((char *) &imgOut[0]);
    } // end for over length
}

// end of file
