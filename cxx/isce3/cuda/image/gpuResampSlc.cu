#include "gpuResampSlc.h"

#include <cmath>

// isce3::core
#include <isce3/core/Constants.h>
#include <isce3/core/Poly2d.h>

#include <isce3/image/Tile.h>
#include <isce3/io/Raster.h>

// isce3::cuda::core
#include <isce3/cuda/core/gpuPoly2d.h>
#include <isce3/cuda/core/gpuLUT1d.h>
#include <isce3/cuda/core/gpuInterpolator.h>

#include <isce3/cuda/except/Error.h>

#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace isce3::cuda::image {

using isce3::cuda::core::gpuPoly2d;
using isce3::cuda::core::gpuInterpolator;
using isce3::cuda::core::gpuLUT1d;
using isce3::cuda::core::gpuSinc2dInterpolator;

#define THRD_PER_BLOCK 512// Number of threads per block (should always %32==0)

/** Transform the original tile by sinc interpolation.
  *
  * \param[out] resampledSlc    Pointer to where transformed SLC data is to be written.
  * \param[in]  originalTile    Pointer to original untransformed SLC data.
  * \param[in]  chip            Pointer to array where interpolation chips are stored.
  * \param[in]  rgOffTile       Pointer to range offsets used in transformation.
  * \param[in]  azOffTile       Pointer to azimuth offsets used in transformation.
  * \param[in]  rgCarrier       Range carrier to be used in phase computation.
  * \param[in]  azCarrier       Azimuth carrier to be used in phase computation.
  * \param[in]  dopplerLUT      Doppler LUT used in phase computation.
  * \param[in]  interp          Interpolator used to transform SLC data.
  * \param[in]  flatten         If true, apply geometric flattening.
  * \param[in]  outWidth        Width of transformed SLC data.
  * \param[in]  outLength       Length of transformed SLC data.
  * \param[in]  inWidth         Width of original SLC data.
  * \param[in]  inReadableLength    Length of readble original SLC data where buffer for chip and offsets are unaccounted.
  * \param[in]  startingRange       Starting range of the resampled radar grid.
  * \param[in]  rangePixelSpacing   Range pixel spacing of the resampled radar grid.
  * \param[in]  sensingStart        Starting azimuth time of the resampled radar grid.
  * \param[in]  prf                 Pulse repition frequency of the radar grid.
  * \param[in]  wavelength          Wavelength of the resampled radar grid.
  * \param[in]  refStartingRange    Staring range of the original/reference radar grid.
  * \param[in]  refRangePixelSpacing    Range pixel spacing of the original/reference radar grid.
  * \param[in]  refWavelength   Wavelength of the original/reference radar grid.
  * \param[in]  chipSize    Length and width of chip used for sinc interpolation.
  * \param[in]  rowOffset   Offset to account for difference between first row of original tile and tile with buffer.
  * \param[in]  rowStart    First row of original tile without buffer accounting for offset and chip.
  * \param[in]  invalid_value   Default value of pixel. Will be overwritten by resampled value if pixel is valid.
*/
__global__
void transformTile(thrust::complex<float> *resampledSlc,
                   const thrust::complex<float> *originalTile,
                   thrust::complex<float> *chip,
                   const double *rgOffTile,
                   const double *azOffTile,
                   const gpuPoly2d rgCarrier,
                   const gpuPoly2d azCarrier,
                   const gpuLUT1d<double> dopplerLUT,
                   gpuSinc2dInterpolator<thrust::complex<float>> interp,
                   bool flatten,
                   size_t outWidth,
                   size_t outLength,
                   size_t inWidth,
                   size_t inReadableLength,
                   double startingRange,
                   double rangePixelSpacing,
                   double sensingStart,
                   double prf,
                   double wavelength,
                   double refStartingRange,
                   double refRangePixelSpacing,
                   double refWavelength,
                   int chipSize,
                   int rowOffset,
                   int rowStart,
                   const thrust::complex<float> invalid_value)
{
    const long long int iTileOut = blockDim.x * blockIdx.x + threadIdx.x;
    const long long int iChip = iTileOut * chipSize * chipSize;
    const int chipHalf = chipSize / 2;

    if (iTileOut < outWidth * outLength)
    {
        // Compute row and column of current tile pixel to be resampled.
        long long int iRow = iTileOut / outWidth;
        long long int iCol = iTileOut % outWidth;

        // Initialize current pixel to invalid value to be overwritten with
        // resampled value if boundary and LUT2d conditions met.
        resampledSlc[iTileOut] = invalid_value;

        // Retreive offsets for current pixel.
        const double azOff = azOffTile[iTileOut];
        const double rgOff = rgOffTile[iTileOut];

        // Compute row and column where current tile pixel is to be resampled.
        // * Row and column are whole numbers.
        // * Add azimuth and range offsets to row and column respectively.
        // * Since blocking is performed along rows, add an offset to the row
        //   index to account for difference in number of rows in tile padded
        //   to accomodate chip used in sinc interpolation.
        // * No offset needed for column as block processing along rows only.
        const auto iRowResamp = iRow + __double2ll_rn(azOff) + rowOffset;
        const auto iColResamp = iCol + __double2ll_rn(rgOff);

        // Compute decimal remainders of row and column.
        // Non-double operands below are promoted to double.
        const double iRowResampFrac = iRow + azOff - iRowResamp + rowOffset;
        const double iColResampFrac = iCol + rgOff - iColResamp;

        // Check if resampled row index is in bounds by checking if chip used
        // to resample can be populated.
        const bool rowOutOfBounds =
            // Check if resampling possible at the starting rows of a tile.
            (iRowResamp < chipHalf)
            // Check if resampling possible at the ending rows of a tile.
            || (iRowResamp + chipHalf > inReadableLength);

        // Check if resampled column index is in bounds by checking if chip
        // used to resample can be populated.
        const bool colOutOfBounds =
            // Check if resampling possible at starting columns of a tile.
            (iColResamp - chipHalf < 0)
            // Check if resampling possible at the ending columns of a tile.
            || (iColResamp + chipHalf > inWidth);

        // Skip computations if indices out of bound or az/rng not in doppler
        if (rowOutOfBounds || colOutOfBounds || iRowResamp < 0 || iColResamp < 0)
            return;

        // Compute azimuth time at iRow index + azimuth offset. Unlike other,
        // this computation needs to be computed w.r.t. raster coordinates.
        const double az = sensingStart + (iRow + rowStart) / prf;

        // Slant range at iCol index + range offset
        const double rng = startingRange + iCol * rangePixelSpacing;

        // evaluate Doppler polynomial
        const double dop = dopplerLUT.eval(rng) * 2 * M_PI / prf;

        // Doppler to be added back. Simultaneously evaluate carrier that needs
        // to be added back after interpolation
        // Account for resample decimal remainder offsets later in carrier
        // compuations.
        const double azPlusOffset = az
            + static_cast<double>(azOff) / prf;
        const double rngPlusOffset = rng
            + static_cast<double>(rgOff) * rangePixelSpacing;
        double phase = (dop * iRowResampFrac)
            + rgCarrier.eval(azPlusOffset, rngPlusOffset)
            + azCarrier.eval(azPlusOffset, rngPlusOffset);

        // Flatten the carrier phase if requested
        if (flatten) {
            phase += ((4. * (M_PI / wavelength)) *
                ((startingRange - refStartingRange)
                + (iCol * (rangePixelSpacing - refRangePixelSpacing))
                + (rgOff * rangePixelSpacing))) + ((4.0 * M_PI
                * (refStartingRange + (iCol * refRangePixelSpacing)))
                * ((1.0 / refWavelength) - (1.0 / wavelength)));
        }

        // Read data chip without the carrier phases
        for (int iChipRow = 0; iChipRow < chipSize; ++iChipRow) {
            // Row in original tile to read from
            const long long int iTileRow = iRowResamp + iChipRow - chipHalf;
            // Carrier phase
            const double phase = dop * (iChipRow - chipHalf);
            const thrust::complex<float> cval(cos(phase), -sin(phase));

            // Set the data values after removing doppler in azimuth
            for (int iChipCol = 0; iChipCol < chipSize; ++iChipCol) {
                // Column in tile to read from
                const long long int iTileCol = iColResamp + iChipCol - chipHalf;
                chip[iChip + iChipRow * chipSize + iChipCol] = originalTile[iTileRow * inWidth + iTileCol] * cval;
            }
        }

        // Interpolate chip
        const thrust::complex<float> cval = interp.interpolate(
            chipHalf + iColResampFrac,
            chipHalf + iRowResampFrac,
            &chip[iChip],
            chipSize,
            chipSize
        );

        // Add doppler to interpolated value and save
        resampledSlc[iTileOut] = cval * thrust::complex<float>(cos(phase), sin(phase));
    }
}


// Interpolate tile to perform transformation
void
gpuTransformTile(
        isce3::io::Raster & outputSlc,
        isce3::image::Tile<std::complex<float>> & origSlcTile,
        isce3::image::Tile<double> & rgOffTile,
        isce3::image::Tile<double> & azOffTile,
        const isce3::core::Poly2d & rgCarrier,
        const isce3::core::Poly2d & azCarrier,
        const isce3::core::LUT1d<double> & dopplerLUT,
        isce3::cuda::core::gpuSinc2dInterpolator<thrust::complex<float>> interp,
        size_t inWidth,
        size_t inLength,
        double startingRange,
        double rangePixelSpacing,
        double sensingStart,
        double prf,
        double wavelength,
        double refStartingRange,
        double refRangePixelSpacing,
        double refWavelength,
        bool flatten,
        int chipSize,
        const std::complex<float> invalid_value) {

    // Cache geometry values
    const size_t outWidth = azOffTile.width();
    const size_t outLength = azOffTile.length();

    // Allocate valarray for output image block
    // Initialize/fill with invalid values
    thrust::host_vector<std::complex<float>> h_resampledSlc(outLength * outWidth,
            invalid_value);

    // determine sizes
    const size_t nOutPixels = h_resampledSlc.size();
    const size_t numelChip = nOutPixels * chipSize * chipSize;

    // declare equivalent objects in device memory
    thrust::device_vector<thrust::complex<float>> d_origSlcTile(origSlcTile.data().size());
    thrust::device_vector<thrust::complex<float>> d_chip(numelChip);
    thrust::device_vector<thrust::complex<float>> d_resampledSlc(h_resampledSlc.size());
    thrust::device_vector<double> d_rgOffTile(rgOffTile.data().size());
    thrust::device_vector<double> d_azOffTile(azOffTile.data().size());
    gpuPoly2d d_rgCarrier(rgCarrier);
    gpuPoly2d d_azCarrier(azCarrier);
    gpuLUT1d<double> d_dopplerLUT(dopplerLUT);

    // copy objects to device memory
    thrust::copy(std::begin(origSlcTile.data()),
            std::end(origSlcTile.data()),
            d_origSlcTile.begin());
    thrust::copy(std::begin(rgOffTile.data()),
            std::end(rgOffTile.data()),
            d_rgOffTile.begin());
    thrust::copy(std::begin(azOffTile.data()),
            std::end(azOffTile.data()),
            d_azOffTile.begin());

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((nOutPixels+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // Compute offset that accounts for difference between first rows of
    // original tile and tile with buffer. Buffer added to account for offset
    // and chip.
    const auto rowReadOffset = origSlcTile.rowStart() - origSlcTile.firstImageRow();

    // First row of original tile without buffer accounting for offset and chip.
    const auto rowStart = origSlcTile.rowStart();

    // Set convert std::complex to thurst::complex for invalid value.
    const thrust::complex d_invalid_value(invalid_value);

    // global call to transform
    transformTile<<<grid, block>>>(d_resampledSlc.data().get(),
                                   d_origSlcTile.data().get(),
                                   d_chip.data().get(),
                                   d_rgOffTile.data().get(),
                                   d_azOffTile.data().get(),
                                   d_rgCarrier,
                                   d_azCarrier,
                                   d_dopplerLUT,
                                   interp,
                                   flatten,
                                   outWidth,
                                   outLength,
                                   origSlcTile.width(),
                                   origSlcTile.length(),
                                   startingRange,
                                   rangePixelSpacing,
                                   sensingStart,
                                   prf,
                                   wavelength,
                                   refStartingRange,
                                   refRangePixelSpacing,
                                   refWavelength,
                                   chipSize,
                                   rowReadOffset,
                                   rowStart,
                                   d_invalid_value);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // copy to host memory
    h_resampledSlc = d_resampledSlc;

    // Write block of data
    outputSlc.setBlock(&h_resampledSlc[0], 0, origSlcTile.rowStart(), outWidth, outLength);
}

} // namespace isce3::cuda::image
