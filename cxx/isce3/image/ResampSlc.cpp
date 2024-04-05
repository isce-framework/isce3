#include "ResampSlc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include <pyre/journal.h>

#include <isce3/core/Constants.h>

#include "Tile.h"

namespace isce3 { namespace image {

using isce3::io::Raster;

// Alternative generic resamp entry point: use filenames to internally create
// rasters
void ResampSlc::resamp(
        const std::string& inputFilename,    // filename of input SLC
        const std::string& outputFilename,   // filename of output resampled SLC
        const std::string& rgOffsetFilename, // filename of range offsets
        const std::string& azOffsetFilename, // filename of azimuth offsets
        int inputBand, bool flatten, int rowBuffer,
        int chipSize)
{
    // Make input rasters
    Raster inputSlc(inputFilename, GA_ReadOnly);
    Raster rgOffsetRaster(rgOffsetFilename, GA_ReadOnly);
    Raster azOffsetRaster(azOffsetFilename, GA_ReadOnly);

    // Make output raster; geometry defined by offset rasters
    const size_t outLength = rgOffsetRaster.length();
    const size_t outWidth = rgOffsetRaster.width();
    Raster outputSlc(outputFilename, outWidth, outLength, 1, GDT_CFloat32,
                     "ISCE");

    // Call generic resamp
    resamp(inputSlc, outputSlc, rgOffsetRaster, azOffsetRaster, inputBand,
           flatten, rowBuffer, chipSize);
}

// Generic resamp entry point from externally created rasters
void ResampSlc::resamp(isce3::io::Raster& inputSlc,
                       isce3::io::Raster& outputSlc,
                       isce3::io::Raster& rgOffsetRaster,
                       isce3::io::Raster& azOffsetRaster,
                       int inputBand,
                       bool flatten,
                       int rowBuffer,
                       int chipSize)
{
    // Set the band number for input SLC
    _inputBand = inputBand;
    // Cache width of SLC image
    const size_t inLength = inputSlc.length();
    const size_t inWidth = inputSlc.width();
    // Cache output length and width from offset images
    const size_t outLength = rgOffsetRaster.length();
    const size_t outWidth = rgOffsetRaster.width();

    // Initialize resampling methods
    _prepareInterpMethods(isce3::core::SINC_METHOD, chipSize - 1);

    // Determine number of tiles needed to process image
    const size_t nTiles = _computeNumberOfTiles(outLength, _linesPerTile);
    std::cout << "Resampling using " << nTiles << " tiles of " << _linesPerTile
              << " lines per tile\n";
    // Start timer
    auto timerStart = std::chrono::steady_clock::now();

    // For each full tile of _linesPerTile lines...
    for (size_t tileCount = 0; tileCount < nTiles; tileCount++) {

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
        Tile<double> azOffTile, rgOffTile;
        _initializeOffsetTiles(tile, azOffsetRaster, rgOffsetRaster, azOffTile,
                               rgOffTile, outWidth);

        // Get corresponding image tile with read extents adjusted for offsets
        // sinc interpolation and chip.
        std::cout << "Reading in image data for tile " << tileCount << "\n";
        _initializeTile(tile, inputSlc, azOffTile, outLength, rowBuffer,
                        chipSize / 2);

        // Perform interpolation
        std::cout << "Interpolating tile " << tileCount << "\n";
        _transformTile(tile, outputSlc, rgOffTile, azOffTile, inLength, flatten,
                       chipSize);
    }

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed =
            1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
                             timerEnd - timerStart)
                             .count();
    std::cout << "Elapsed processing time: " << elapsed << " sec\n";
}

// Initialize and read azimuth and range offsets
void ResampSlc::_initializeOffsetTiles(Tile_t& tile, Raster& azOffsetRaster,
                                       Raster& rgOffsetRaster,
                                       Tile<double>& azOffTile,
                                       Tile<double>& rgOffTile, size_t outWidth)
{
    // Copy size properties and initialize azimuth offset tiles
    azOffTile.width(outWidth);
    azOffTile.rowStart(tile.rowStart());
    azOffTile.rowEnd(tile.rowEnd());
    azOffTile.firstImageRow(tile.rowStart());
    azOffTile.lastImageRow(tile.rowEnd());
    azOffTile.allocate();

    // Do the same for range offset tiles
    rgOffTile.width(outWidth);
    rgOffTile.rowStart(tile.rowStart());
    rgOffTile.rowEnd(tile.rowEnd());
    rgOffTile.firstImageRow(tile.rowStart());
    rgOffTile.lastImageRow(tile.rowEnd());
    rgOffTile.allocate();

    // Read in block of range and azimuth offsets
    azOffsetRaster.getBlock(&azOffTile[0], 0, azOffTile.rowStart(),
                            azOffTile.width(), azOffTile.length());
    rgOffsetRaster.getBlock(&rgOffTile[0], 0, rgOffTile.rowStart(),
                            rgOffTile.width(), rgOffTile.length());
}

// Initialize tile bounds
void ResampSlc::_initializeTile(Tile_t& tile, Raster& inputSlc,
                                const Tile<double>& azOffTile, size_t outLength,
                                int rowBuffer, int chipHalf)
{
    // Cache geometry values
    const size_t inLength = inputSlc.length();
    const size_t inWidth = inputSlc.width();
    const size_t outWidth = azOffTile.width();

    // Convert to double and cache values for later use.
    const auto azOffTileRowStartDbl = static_cast<double>(azOffTile.rowStart());
    const auto chipHalfDbl = static_cast<double>(chipHalf);

    // Compute minimum read-from-raster row index needed from input image.
    tile.firstImageRow(outLength - 1);
    bool haveOffsets = false;
    for (size_t iRow = 0;
         iRow < std::min(static_cast<size_t>(rowBuffer), azOffTile.length());
         ++iRow) {
        for (size_t iCol = 0; iCol < outWidth; ++iCol) {
            // Get azimuth offset for pixel
            const double azOff = azOffTile(iRow, iCol);
            // Skip null values
            if (azOff < -5.0e5 || std::isnan(azOff)) {
                continue;
            } else {
                haveOffsets = true;
            }

            // imageLine is zero if difference is less than zero i.e. underflow
            // Round offset to get to nearest pixel center
            // "+ azOffTile.rowStart()" puts resulting index in reference to the raster.
            // "- chipHalf" accounts for leading row of the tile used to populate of the associated chip.
            const auto imageLineDbl = static_cast<double>(iRow)
                + std::round(azOff) + azOffTileRowStartDbl - chipHalfDbl;
            const size_t imageLine = imageLineDbl < 0 ?
                0 : static_cast<size_t>(imageLineDbl);

            // Update minimum row index
            tile.firstImageRow(std::min(tile.firstImageRow(), imageLine));
        }
    }
    // Final update
    if (haveOffsets) {
        tile.firstImageRow(std::max(tile.firstImageRow(), static_cast<size_t>(0)));
    } else {
        tile.firstImageRow(0);
    }

    // Compute maximum read-from-raster row index needed from input image.
    tile.lastImageRow(0);
    haveOffsets = false;
    for (size_t iRow = std::max(azOffTile.length() - rowBuffer, static_cast<size_t>(0));
         iRow < azOffTile.length(); ++iRow) {
        for (size_t iCol = 0; iCol < outWidth; ++iCol) {
            // Get azimuth offset for pixel
            const double azOff = azOffTile(iRow, iCol);
            // Skip null values
            if (azOff < -5.0e5 || std::isnan(azOff)) {
                continue;
            } else {
                haveOffsets = true;
            }

            // imageLine is zero if difference is less than zero i.e. underflow
            // Round offset to get to nearest pixel center
            // "+ azOffTile.rowStart()" puts resulting index in reference to the raster.
            // "+ chipHalf" accounts for trailing row of the tile used to populate of the associated chip.
            const auto imageLineDbl = static_cast<double>(iRow)
                + std::round(azOff) + azOffTileRowStartDbl + chipHalfDbl;
            const size_t imageLine = imageLineDbl < 0 ?
                0 : static_cast<size_t>(imageLineDbl);

            // Update maximum row index
            tile.lastImageRow(std::max(tile.lastImageRow(), imageLine));
        }
    }
    // Final udpate
    if (haveOffsets) {
        // +1 to account for computed lastImageRow being a valid index so
        // length() in Tile is correctly computed.
        tile.lastImageRow(std::min(tile.lastImageRow() + 1, inLength));
    } else {
        tile.lastImageRow(inLength);
    }

    // Tile will allocate memory for itself
    tile.allocate();

    // Read in tile.length() lines of data from the input image to the image
    // block
    inputSlc.getBlock(&tile[0], 0, tile.firstImageRow(), tile.width(),
                      tile.length(), _inputBand);

    // Remove carrier from input data
    for (size_t iRow = 0; iRow < tile.length(); iRow++) {
        const double az =  _sensingStart + (iRow + tile.firstImageRow()) / _prf;
        for (size_t iCol = 0; iCol < inWidth; iCol++) {
            const double rng = _startingRange + iCol * _rangePixelSpacing;
            // Evaluate the pixel's carrier phase
            const double phase = _rgCarrier.eval(az, rng)
                + _azCarrier.eval(az, rng);
            // Remove the carrier
            std::complex<float> cpxPhase(std::cos(phase), -std::sin(phase));
            tile(iRow, iCol) *= cpxPhase;
        }
    }
}

// Interpolate tile to perform transformation
void ResampSlc::_transformTile(Tile_t& originalTile, Raster& outputSlc,
                               const Tile<double>& rgOffTile,
                               const Tile<double>& azOffTile, size_t inLength,
                               bool flatten, int chipSize)
{
    if (flatten && !_haveRefData) {
        std::string error_msg{"Unable to flatten; reference data not provided."};
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    // Cache geometry values
    const size_t inWidth = originalTile.width();
    const size_t outWidth = azOffTile.width();
    const size_t outLength = azOffTile.length();
    int chipHalf = chipSize / 2;

    // Allocate valarray for resampled output image block
    std::valarray<std::complex<float>> resampledTile(outLength * outWidth);
    // Initialize/fill with invalid values
    resampledTile = _invalid_value;

    // From this point on, transformation is multithreaded
    size_t tileLine = 0;
    _Pragma("omp parallel shared(resampledTile)")
    {

        // set half chip size
        // Allocate matrix for working sinc chip
        isce3::core::Matrix<std::complex<float>> chip(chipSize, chipSize);

        for (size_t iRow = originalTile.rowStart(); iRow < originalTile.rowEnd(); ++iRow)
        {

            // Compute azimuth time at iRow index
            const double az = _sensingStart + iRow / _prf;

            // Cache double casted row index.
            const auto iRowDbl = static_cast<double>(iRow);

            // Loop over width
            _Pragma("omp for") for (size_t iCol = 0; iCol < outWidth; ++iCol)
            {

                // Unpack offsets (units of bins)
                const double azOff = azOffTile(tileLine, iCol);
                const double rgOff = rgOffTile(tileLine, iCol);

                // Round offset to get to nearest pixel center and add to
                // original indices to convert to resampled indices. Save as
                // double. If result negative, out of bounds pixel encountered
                // and skip further processing. After addition, compute
                // fractional remainder of azimuth and range offset-adjusted
                // index.
                const auto iRowResampledDbl = iRowDbl + std::round(azOff);
                if (iRowResampledDbl < 0)
                    continue;
                const auto iRowResampled = static_cast<size_t>(iRowResampledDbl);
                // Both size_t operands below are promoted to double.
                const double fracAz = iRowDbl + azOff - iRowResampledDbl;

                const auto iColResampledDbl =
                    static_cast<double>(iCol) + std::round(rgOff);
                if (iColResampledDbl < 0)
                    continue;
                const auto iColResampled = static_cast<size_t>(iColResampledDbl);
                // Both size_t operands below are promoted to double.
                const double fracRg = iCol + rgOff - iColResampledDbl;

                // Check bounds
                if ((iRowResampled < chipHalf) ||
                        (iRowResampled >= (inLength - chipHalf)))
                    continue;
                if ((iColResampled < chipHalf) || (
                            iColResampled >= (inWidth - chipHalf)))
                    continue;

                // Slant range at iCol index
                const double rng = _startingRange + iCol * _rangePixelSpacing;

                // Check if the Doppler LUT covers the current position
                if (not _dopplerLUT.contains(az, rng))
                    continue;

                // Evaluate Doppler polynomial
                const double dop = _dopplerLUT.eval(az, rng) * 2 * M_PI / _prf;

                // Doppler to be added back. Simultaneously evaluate carrier
                // that needs to be added back after interpolation
                // Account for resample offsets in carrier evaluations.
                const double azPlusOffset = az
                    + static_cast<double>(azOff) / _prf;
                const double rngPlusOffset = rng
                    + static_cast<double>(rgOff) * _rangePixelSpacing;
                double phase = (dop * fracAz)
                    + _rgCarrier.eval(azPlusOffset, rngPlusOffset)
                    + _azCarrier.eval(azPlusOffset, rngPlusOffset);

                // Flatten the carrier phase if requested
                if (flatten && _haveRefData) {
                    phase += ((4. * (M_PI / _wavelength)) *
                              ((_startingRange - _refStartingRange) +
                               (iCol *
                                (_rangePixelSpacing - _refRangePixelSpacing)) +
                               (rgOff * _rangePixelSpacing))) +
                             ((4.0 * M_PI *
                               (_refStartingRange +
                                (iCol * _refRangePixelSpacing))) *
                              ((1.0 / _refWavelength) - (1.0 / _wavelength)));
                }

                // Read data chip without the carrier phases
                for (int iChipRow = 0; iChipRow < chipSize; ++iChipRow) {
                    // Row to read from
                    const int iChipRowInTile =
                            iRowResampled - originalTile.firstImageRow() + iChipRow - chipHalf;
                    // Carrier phase
                    const double chipPhase = dop * (iChipRow - chipHalf);
                    const std::complex<float> cval(
                            std::cos(chipPhase), -std::sin(chipPhase));
                    // Set the data values after removing doppler in azimuth
                    for (int iChipCol = 0; iChipCol < chipSize; ++iChipCol) {
                        // Column to read from
                        const int iChipColInTile = iColResampled + iChipCol - chipHalf;
                        chip(iChipRow, iChipCol) =
                            originalTile(iChipRowInTile, iChipColInTile) * cval;
                    }
                }

                // Interpolate chip
                const std::complex<float> cval = _interp->interpolate(
                        chipHalf + fracRg, chipHalf + fracAz, chip);

                // Add doppler to interpolated value and save to resampled tile.
                resampledTile[tileLine * outWidth + iCol] =
                        cval *
                        std::complex<float>(std::cos(phase), std::sin(phase));

            } // end for over width

            // Update input line counter
            _Pragma("omp single") { ++tileLine; }

        } // end for over length

    } // end multithreaded block

    // Write block of data
    outputSlc.setBlock(resampledTile, 0, originalTile.rowStart(), outWidth, outLength);
}

}} // namespace isce3::image
