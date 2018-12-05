//-*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright: 2018

// isce::core
#include "isce/core/Interpolator.h"
#include "isce/core/Poly2d.h"

// isce::cuda::core
#include "isce/cuda/core/gpuComplex.h"
#include "isce/cuda/core/gpuInterpolator.h"
#include "isce/cuda/core/gpuPoly2d.h"

// isce::cuda::image
#include "gpuResampSlc.h"

#include "isce/cuda/helper_cuda.h"

using isce::cuda::core::gpuComplex;

/*
__global__
void transformTile() (tile,
                    imgOut,
                    const & rgOffTile,
                    const & azOffTile,
                    const isce::core::Poly2d & rgCarrier,
                    const isce::core::Poly2d & azCarrier,
                    const isce::core::Poly2d & doppler,
                    int inLength, bool flatten) {
}
*/

// Interpolate tile to perform transformation
void isce::cuda::image::
gpuTransformTile(isce::image::Tile<std::complex<float>> & tile,
               isce::io::Raster & outputSlc,
               isce::image::Tile<float> & rgOffTile,
               isce::image::Tile<float> & azOffTile,
               const isce::core::Poly2d & rgCarrier,
               const isce::core::Poly2d & azCarrier,
               const isce::core::Poly2d & doppler,
               int inLength, bool flatten) {

    // Cache geometry values
    const int inWidth = tile.width();
    const int outWidth = azOffTile.width();
    const int outLength = azOffTile.length();

    // Allocate valarray for output image block
    std::valarray<std::complex<float>> imgOut(outLength * outWidth);
    // Initialize to zeros
    imgOut = std::complex<float>(0.0, 0.0);

    // From this point on, transformation is multithreaded
    int tileLine = 0;

    // declare equivalent objects in device memory
    gpuComplex<float> *dTile, *dImgOut;
    float *dRgOffTile, *dAzOffTile;
    isce::cuda::core::gpuPoly2d dRgCarrier(rgCarrier);
    isce::cuda::core::gpuPoly2d dAxCarrier(azCarrier);
    isce::cuda::core::gpuPoly2d dDoppler(doppler);

    // allocate equivalent ofjects in device memory
    // gpuPoly2d objects allocated when instantiated
    checkCudaErrors(cudaMalloc(&dTile, tile.length()*tile.width()*sizeof(gpuComplex<float>)));
    checkCudaErrors(cudaMalloc(&dImgOut, imgOut.size()*sizeof(gpuComplex<float>)));
    checkCudaErrors(cudaMalloc(&dAzOffTile, azOffTile.length()*azOffTile.width()*sizeof(float)));
    checkCudaErrors(cudaMalloc(&dRgOffTile, rgOffTile.length()*rgOffTile.width()*sizeof(float)));

    // copy equivalent objects to device memory
    checkCudaErrors(cudaMemcpy(dTile, &tile.data()[0], tile.length()*tile.width()*sizeof(gpuComplex<float>), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dAzOffTile, &azOffTile.data()[0], azOffTile.length()*azOffTile.width()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dRgOffTile, &rgOffTile.data()[0], rgOffTile.length()*rgOffTile.width()*sizeof(float), cudaMemcpyHostToDevice));

    // global call to transform

    // copy to host memory
    checkCudaErrors(cudaMemcpy(&imgOut[0], dImgOut, imgOut.size()*sizeof(gpuComplex<float>), cudaMemcpyDeviceToHost));

    // deallocate to device memory
    checkCudaErrors(cudaFree(dTile));
    checkCudaErrors(cudaFree(dImgOut));
    checkCudaErrors(cudaFree(dAzOffTile));
    checkCudaErrors(cudaFree(dRgOffTile));

    /*
    // Allocate matrix for working sinc chip
    isce::core::Matrix<std::complex<float>> chip(SINC_ONE, SINC_ONE);
    
    // Loop over lines to perform interpolation
    for (int i = tile.rowStart(); i < tile.rowEnd(); ++i) {

        // Loop over width
        for (int j = 0; j < outWidth; ++j) {

            // Unpack offsets
            const float azOff = azOffTile(tileLine, j);
            const float rgOff = rgOffTile(tileLine, j);

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
            const double dop = _dopplerPoly.eval(0, j) * 2*M_PI / _mode.prf();

            // Doppler to be added back. Simultaneously evaluate carrier that needs to
            // be added back after interpolation
            double phase = (dop * fracAz) 
                + _rgCarrier.eval(i + azOff, j + rgOff) 
                + _azCarrier.eval(i + azOff, j + rgOff);

            // Flatten the carrier phase if requested
            if (flatten && _haveRefMode) {
                phase += ((4. * (M_PI / _mode.wavelength())) * 
                    ((_mode.startingRange() - _refMode.startingRange()) 
                    + (j * (_mode.rangePixelSpacing() - _refMode.rangePixelSpacing())) 
                    + (rgOff * _mode.rangePixelSpacing()))) + ((4.0 * M_PI 
                    * (_refMode.startingRange() + (j * _refMode.rangePixelSpacing()))) 
                    * ((1.0 / _refMode.wavelength()) - (1.0 / _mode.wavelength())));
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

            // Interpolate chip
            const std::complex<float> cval = _interp->interpolate(
                SINC_HALF + fracRg + 1, SINC_HALF + fracAz + 1, chip
            );

            // Add doppler to interpolated value and save
            imgOut[tileLine*outWidth + j] = cval * std::complex<float>(
                std::cos(phase), std::sin(phase)
            );

        } // end for over width

        // Update input line counter
        #pragma omp single
        {
        ++tileLine;
        }

    } // end for over length
    */

    // Write block of data
    outputSlc.setBlock(imgOut, 0, tile.rowStart(), outWidth, outLength);
}
