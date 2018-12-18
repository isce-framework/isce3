//-*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright: 2018

// isce::core
#include "isce/core/Constants.h"
#include "isce/core/Interpolator.h"
#include "isce/core/Poly2d.h"

// isce::cuda::core
#include "isce/cuda/core/gpuComplex.h"
#include "isce/cuda/core/gpuInterpolator.h"
#include "isce/cuda/core/gpuPoly2d.h"

// isce::cuda::image
#include "gpuResampSlc.h"
#include "gpuImageMode.h"

#include "isce/cuda/helper_cuda.h"

using isce::cuda::core::gpuComplex;
using isce::cuda::core::gpuPoly2d;
using isce::cuda::core::gpuInterpolator;
using isce::cuda::core::gpuSinc2dInterpolator;
using isce::cuda::image::gpuImageMode;

#define THRD_PER_BLOCK 16 // Number of threads per block (should always %32==0)
#define SINC_ONE 9
#define SINC_HALF 4

__global__
void transformTile(const gpuComplex<float> *tile,
                   gpuComplex<float> *imgOut,
                   const float *rgOffTile,
                   const float *azOffTile,
                   const gpuPoly2d rgCarrier,
                   const gpuPoly2d azCarrier,
                   const gpuPoly2d doppler,
                   gpuImageMode mode,       // image mode for image to be resampled
                   gpuImageMode refMode,    // image mode for reference master image
                   gpuSinc2dInterpolator<gpuComplex<float>> *interp,
                   bool flatten,
                   int outWidth,
                   int outLength) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    printf("bdx%d, bix%d, tix%d\n", blockDim.x, blockIdx.x, threadIdx.x);
    printf("bdy%d, biy%d, tiy%d\n", blockDim.y, blockIdx.y, threadIdx.y);
    // check bounds
    if (i < outWidth && j < outLength) {
        int iArr = outWidth*i + j;

        gpuComplex<float> chip[SINC_ONE*SINC_ONE];

        // Unpack offsets
        const float azOff = azOffTile[iArr];
        const float rgOff = rgOffTile[iArr];
        printf("azOff %f | rgOff %f\n", azOff, rgOff);

        // Break into fractional and integer parts
        const int intAz = __float2int_rd(i + azOff);
        const int intRg = __float2int_rd(j + rgOff);
        const double fracAz = i + azOff - intAz;
        const double fracRg = j + rgOff - intRg;
       
        // Check bounds again
        bool intAzInBounds = !((intAz < SINC_HALF) || (intAz >= (outLength - SINC_HALF)));
        bool intRgInBounds = !((intRg < SINC_HALF) || (intRg >= (outWidth - SINC_HALF)));

        if (intAzInBounds && intRgInBounds) {
            // evaluate Doppler polynomial
            const double dop = doppler.eval(0, j) * 2 * M_PI / mode.prf;
            printf("dop %f\n", dop);

            // Doppler to be added back. Simultaneously evaluate carrier that needs to
            // be added back after interpolation
            double phase = (dop * fracAz) 
                + rgCarrier.eval(i + azOff, j + rgOff) 
                + azCarrier.eval(i + azOff, j + rgOff);

            printf("phase %f\n", phase);
            printf("azCarrier %f | rgCarrier %f\n", azCarrier.eval(i + azOff, j + rgOff), 
                                                    rgCarrier.eval(i + azOff, j + rgOff));
            // Flatten the carrier phase if requested
            if (flatten && refMode.isRefMode) {
                phase += ((4. * (M_PI / mode.wavelength)) * 
                    ((mode.startingRange - refMode.startingRange) 
                    + (j * (mode.rangePixelSpacing - refMode.rangePixelSpacing)) 
                    + (rgOff * mode.rangePixelSpacing))) + ((4.0 * M_PI 
                    * (refMode.startingRange + (j * refMode.rangePixelSpacing))) 
                    * ((1.0 / refMode.wavelength) - (1.0 / mode.wavelength)));
            }
            printf("phase %f\n", phase);
            // Modulate by 2*PI
            phase = fmod(phase, 2.0*M_PI);
            printf("phase %f\n", phase);
            
            // Read data chip without the carrier phases
            for (int ii = 0; ii < SINC_ONE; ++ii) {
                // Row to read from
                const int chipRow = intAz + ii - SINC_HALF;
                // Carrier phase
                const double phase = dop * (ii - 4.0);
                const gpuComplex<float> cval(cos(phase), -sin(phase));
                // Set the data values after removing doppler in azimuth
                for (int jj = 0; jj < SINC_ONE; ++jj) {
                    // Column to read from
                    const int chipCol = intRg + jj - SINC_HALF;
                    printf("intAz %d ii %d chipRow %d intRg %d jj %d chipCol %d\n", intAz, ii, chipRow, intRg, jj, chipCol);
                    chip[ii*SINC_ONE+jj] = tile[iArr] * cval;
                }
            }

            // Interpolate chip
            const gpuComplex<float> cval = interp->interpolate(
                SINC_HALF + fracRg + 1, SINC_HALF + fracAz + 1, chip, SINC_ONE, SINC_ONE
            );

            // Add doppler to interpolated value and save
            imgOut[iArr] = cval * gpuComplex<float>(cos(phase), sin(phase));
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
               const isce::core::Poly2d & doppler,
               isce::product::ImageMode mode,       // image mode for image to be resampled
               isce::product::ImageMode refMode,    // image mode for reference master image
               bool haveRefMode,
               int inLength, bool flatten) {

    // Cache geometry values
    const int inWidth = tile.width();
    const int outWidth = azOffTile.width();
    const int outLength = azOffTile.length();

    // Allocate valarray for output image block
    std::valarray<std::complex<float>> imgOut(outLength * outWidth);
    // Initialize to zeros
    imgOut = std::complex<float>(0.0, 0.0);

    // declare equivalent objects in device memory
    gpuComplex<float> *d_tile = NULL;
    gpuComplex<float> *d_imgOut = NULL;
    float *d_rgOffTile, *d_azOffTile;
    gpuPoly2d d_rgCarrier(rgCarrier);
    gpuPoly2d d_azCarrier(azCarrier);
    gpuImageMode d_mode(mode);
    gpuImageMode d_refMode;
    if (haveRefMode)
        gpuImageMode d_mode(refMode);
    gpuPoly2d d_doppler(doppler);
    gpuSinc2dInterpolator<gpuComplex<float>> d_interp(isce::core::SINC_LEN, isce::core::SINC_SUB);

    // allocate equivalent ofjects in device memory
    size_t nPixels = imgOut.size();
    size_t nComplexBytes = nPixels * sizeof(gpuComplex<float>);

    checkCudaErrors(cudaMalloc(&d_tile, nPixels*sizeof(gpuComplex<float>)));
    checkCudaErrors(cudaMalloc(&d_imgOut, nPixels*sizeof(gpuComplex<float>)));
    checkCudaErrors(cudaMalloc(&d_azOffTile, nPixels*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_rgOffTile, nPixels*sizeof(float)));

    // copy objects to device memory
    checkCudaErrors(cudaMemcpy(d_tile, &tile[tile.rowStart()], nPixels*sizeof(gpuComplex<float>), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_azOffTile, &azOffTile[azOffTile.rowStart()], nPixels*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rgOffTile, &rgOffTile[rgOffTile.rowStart()], nPixels*sizeof(float), cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK, THRD_PER_BLOCK);
    const int nBlocks_x = (outWidth + THRD_PER_BLOCK - 1) / THRD_PER_BLOCK;
    const int nBlocks_y = (outLength + THRD_PER_BLOCK - 1) / THRD_PER_BLOCK;
    dim3 grid(nBlocks_x, nBlocks_y);

    // global call to transform
    //transformTile<<<grid,block>>>(d_tile, 
    transformTile<<<dim3(1),dim3(1)>>>(d_tile, 
                                d_imgOut, 
                                d_rgOffTile, 
                                d_azOffTile, 
                                d_rgCarrier, 
                                d_azCarrier, 
                                d_doppler, 
                                d_mode, 
                                d_refMode,
                                &d_interp,
                                flatten,
                                outWidth,
                                outLength); 

    // copy to host memory
    checkCudaErrors(cudaMemcpy(&imgOut[0], d_imgOut, nPixels*sizeof(gpuComplex<float>), cudaMemcpyDeviceToHost));

    // deallocate to device memory
    checkCudaErrors(cudaFree(d_tile));
    checkCudaErrors(cudaFree(d_imgOut));
    checkCudaErrors(cudaFree(d_azOffTile));
    checkCudaErrors(cudaFree(d_rgOffTile));
    
    // Write block of data
    outputSlc.setBlock(imgOut, 0, tile.rowStart(), outWidth, outLength);
}
