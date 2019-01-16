//-*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright: 2018
#include <math.h>

// isce::core
#include "isce/core/Constants.h"
#include "isce/core/Poly2d.h"

// isce::cuda::core
#include "isce/cuda/core/gpuPoly2d.h"
#include "isce/cuda/core/gpuLUT1d.h"

// isce::cuda::image
#include "gpuResampSlc.h"
#include "gpuImageMode.h"

#include "isce/cuda/helper_cuda.h"
#include <fstream>
#include <string>
using isce::cuda::core::gpuComplex;
using isce::cuda::core::gpuPoly2d;
using isce::cuda::core::gpuInterpolator;
using isce::cuda::core::gpuLUT1d;
using isce::cuda::core::gpuSinc2dInterpolator;
using isce::cuda::image::gpuImageMode;

#define THRD_PER_BLOCK 512// Number of threads per block (should always %32==0)

__global__
void transformTile(const gpuComplex<float> *tile,
                   gpuComplex<float> *chip,
                   gpuComplex<float> *imgOut,
                   const float *rgOffTile,
                   const float *azOffTile,
                   const gpuPoly2d rgCarrier,
                   const gpuPoly2d azCarrier,
                   const gpuLUT1d<double> dopplerLUT,
                   gpuImageMode mode,       // image mode for image to be resampled
                   gpuImageMode refMode,    // image mode for reference master image
                   gpuSinc2dInterpolator<gpuComplex<float>> interp,
                   bool flatten,
                   int outWidth,
                   int outLength,
                   int inWidth,
                   int inLength,
                   int chipSize,
                   int rowOffset, 
                   int rowStart) {

    int iTileOut = blockDim.x * blockIdx.x + threadIdx.x;
    int iChip = iTileOut * chipSize * chipSize;                                          
    int chipHalf = chipSize/2;

    if (iTileOut < outWidth*outLength) {
        int i = iTileOut / outWidth;
        int j = iTileOut % outWidth;
        imgOut[iTileOut] = gpuComplex<float>(0., 0.);

        // Unpack offsets
        const float azOff = azOffTile[iTileOut];
        const float rgOff = rgOffTile[iTileOut];

        // Break into fractional and integer parts
        const int intAz = __float2int_rd(i + azOff + rowStart);
        const int intRg = __float2int_rd(j + rgOff);
        const double fracAz = i + azOff - intAz + rowStart;
        const double fracRg = j + rgOff - intRg;
       
        // Check bounds again. Use rowOffset to account tiles where tile.rowStart != tile.firstRowImage
        bool intAzInBounds = !((intAz+rowOffset < chipHalf) || (intAz >= (inLength - chipHalf)));
        bool intRgInBounds = !((intRg < chipHalf) || (intRg >= (inWidth - chipHalf)));

        if (intAzInBounds && intRgInBounds) {
            // evaluate Doppler polynomial
            const double rng = mode.startingRange + j * mode.rangePixelSpacing;
            const double dop = dopplerLUT.eval(rng) * 2 * M_PI / mode.prf;

            // Doppler to be added back. Simultaneously evaluate carrier that needs to
            // be added back after interpolation
            double phase = (dop * fracAz) 
                + rgCarrier.eval(i + azOff, j + rgOff) 
                + azCarrier.eval(i + azOff, j + rgOff);

            // Flatten the carrier phase if requested
            if (flatten && refMode.isRefMode) {
                phase += ((4. * (M_PI / mode.wavelength)) * 
                    ((mode.startingRange - refMode.startingRange) 
                    + (j * (mode.rangePixelSpacing - refMode.rangePixelSpacing)) 
                    + (rgOff * mode.rangePixelSpacing))) + ((4.0 * M_PI 
                    * (refMode.startingRange + (j * refMode.rangePixelSpacing))) 
                    * ((1.0 / refMode.wavelength) - (1.0 / mode.wavelength)));
            }
            
            // Modulate by 2*PI
            phase = fmod(phase, 2.0*M_PI);
            
            // Read data chip without the carrier phases
            for (int ii = 0; ii < chipSize; ++ii) {
                // Row to read from
                const int chipRow = intAz + ii - chipHalf + rowOffset - rowStart;
                // Carrier phase
                const double phase = dop * (ii - 4.0);
                const gpuComplex<float> cval(cos(phase), -sin(phase));
                // Set the data values after removing doppler in azimuth
                for (int jj = 0; jj < chipSize; ++jj) {
                    // Column to read from
                    const int chipCol = intRg + jj - chipHalf;
                    chip[iChip + ii*chipSize+jj] = tile[chipRow*outWidth+chipCol] * cval;
                }
            }

            // Interpolate chip
            const gpuComplex<float> cval = interp.interpolate(
                chipHalf + fracRg, chipHalf + fracAz, &chip[iChip], chipSize, chipSize
            );

            // Add doppler to interpolated value and save
            imgOut[iTileOut] = cval * gpuComplex<float>(cos(phase), sin(phase));
        }
    }
}


// Interpolate tile to perform transformation
void isce::cuda::image::
gpuTransformTile(isce::image::Tile<std::complex<float>> & tile,
               isce::io::Raster & outputSlc,
               isce::image::Tile<float> & rgOffTile,
               isce::image::Tile<float> & azOffTile,
               const isce::core::Poly2d & rgCarrier,
               const isce::core::Poly2d & azCarrier,
               const isce::core::LUT1d<double> & dopplerLUT,
               isce::product::ImageMode mode,       // image mode for image to be resampled
               isce::product::ImageMode refMode,    // image mode for reference master image
               bool haveRefMode,
               isce::cuda::core::gpuSinc2dInterpolator<gpuComplex<float>> interp,
               int inWidth, int inLength, bool flatten, int chipSize) {

    // Cache geometry values
    const int outWidth = azOffTile.width();
    const int outLength = azOffTile.length();

    // Allocate valarray for output image block
    std::valarray<std::complex<float>> imgOut(outLength * outWidth);
    // Initialize to zeros
    imgOut = std::complex<float>(0.0, 0.0);

    // declare equivalent objects in device memory
    gpuComplex<float> *d_tile;
    gpuComplex<float> *d_chip;
    gpuComplex<float> *d_imgOut;
    float *d_rgOffTile, *d_azOffTile;
    gpuPoly2d d_rgCarrier(rgCarrier);
    gpuPoly2d d_azCarrier(azCarrier);
    gpuImageMode d_mode(mode);
    gpuImageMode d_refMode;             // empty by default
    if (haveRefMode)
        gpuImageMode d_mode(refMode);   // populate from CPU version if provided
    gpuLUT1d<double> d_dopplerLUT(dopplerLUT);

    // determine sizes
    size_t nInPixels = (tile.lastImageRow() - tile.firstImageRow() + 1) * outWidth;
    size_t nOutPixels = imgOut.size();
    size_t nOutBytes = nOutPixels * sizeof(gpuComplex<float>);
    size_t nChipBytes = nOutBytes * chipSize * chipSize;

    // allocate equivalent objects in device memory
    checkCudaErrors(cudaMalloc(&d_tile, nInPixels*sizeof(gpuComplex<float>)));
    checkCudaErrors(cudaMalloc(&d_chip, nChipBytes));
    checkCudaErrors(cudaMalloc(&d_imgOut, nOutBytes));
    checkCudaErrors(cudaMalloc(&d_azOffTile, nOutPixels*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_rgOffTile, nOutPixels*sizeof(float)));

    // copy objects to device memory
    checkCudaErrors(cudaMemcpy(d_tile, &tile[0], nInPixels*sizeof(gpuComplex<float>), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_azOffTile, &azOffTile[0], nOutPixels*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rgOffTile, &rgOffTile[0], nOutPixels*sizeof(float), cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((nOutPixels+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // global call to transform
    transformTile<<<grid, block>>>(d_tile, 
                                   d_chip,
                                   d_imgOut, 
                                   d_rgOffTile, 
                                   d_azOffTile, 
                                   d_rgCarrier, 
                                   d_azCarrier, 
                                   d_dopplerLUT, 
                                   d_mode, 
                                   d_refMode,
                                   interp,
                                   flatten,
                                   outWidth,
                                   outLength,
                                   inWidth,
                                   inLength,
                                   chipSize,
                                   tile.rowStart()-tile.firstImageRow(),// needed to keep az in bounds in subtiles
                                   tile.rowStart());                    // needed to match az components on CPU

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());

    // copy to host memory
    checkCudaErrors(cudaMemcpy(&imgOut[0], d_imgOut, nOutBytes, cudaMemcpyDeviceToHost));

    // deallocate to device memory
    checkCudaErrors(cudaFree(d_tile));
    checkCudaErrors(cudaFree(d_chip));
    checkCudaErrors(cudaFree(d_imgOut));
    checkCudaErrors(cudaFree(d_azOffTile));
    checkCudaErrors(cudaFree(d_rgOffTile));
    
    // Write block of data
    outputSlc.setBlock(imgOut, 0, tile.rowStart(), outWidth, outLength);
}
