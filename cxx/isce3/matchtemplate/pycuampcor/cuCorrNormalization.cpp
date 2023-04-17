/*
 * @file cuCorrNormalization.cu
 * @brief Utilities to normalize the correlation surface
 *
 * The mean and variance of the normalization factor can be computed from the
 *   cumulative/prefix sum (or sum area table) s(u,v), and s2(u,v).
 * We follow the algorithm by Evghenii Gaburov, Tim Idzenga, Willem Vermin, in the nxcor package.
 * 1. Iterate over rows and for each row, the cumulative sum for elements in the row
 *    is computed as c_row(u,v) = \sum_(v'<v) f(u, v')
 *    and we keep track of the sum of area of width Ny, i.e.,
 *         c(u,v) = \sum_{u'<=u} [c_row(u', v+Ny) - c_row(u', v)],
 *         or c(u,v) = c(u-1, v) + [c_row(u, v+Ny) - c_row(u, v)]
 * 2. When row reaches the window height u=Nx-1,
 *    c(u,v) provides the sum of area for the first batch of windows sa(0,v).
 * 3. proceeding to row = u+1, we compute both c_row(u+1, v) and c_row(u-Nx, v)
 *    i.e., we add the sum from new row and remove the sum from the first row in c(u,v):
 *    c(u+1,v)= c(u,v) + [c_row(u+1,v+Ny)-c_row(u+1, v)] - [c_row(u-Nx, v+Ny)-c_row(u-Nx, v)].
 * 4. Iterate 3. over the rest rows, and c(u,v) provides the sum of areas for new row of windows.
 *
 */

#include "cuAmpcorUtil.h"

#include <cfloat>
#include <stdio.h>

namespace isce3::matchtemplate::pycuampcor {

/**
 * Compute and subtract mean values from images
 * @param[inout] images Input/Output images
 * @param[out] mean Output mean values
 */
void cuArraysSubtractMean(cuArrays<float> *images)
{
    const int imageSize = images->width*images->height;
    const float invSize = 1.0f/imageSize;

    for (int imageIdx = 0; imageIdx < images->count; imageIdx++) {
        double sum = 0.0f;
        const int imageOffset = imageIdx * imageSize;
        float* imageD = images->devData + imageOffset;
        for (int i = 0; i < imageSize; i++) {
            sum += imageD[i];
        }
        const float mean = sum * invSize;
        for (int i = 0; i < imageSize; i++) {
            imageD[i] -= mean;
        }
    }
}


/**
 * Compute the variance of images (for SNR)
 * @param[in] images Input images
 * @param[in] imagesValid validity flags for each pixel
 * @param[out] imagesSum variance
 * @param[out] imagesValidCount count of total valid pixels
 */
void cuArraysSumCorr(cuArrays<float> *images, cuArrays<int> *imagesValid, cuArrays<float> *imagesSum,
    cuArrays<int> *imagesValidCount)
{
    const int imageSize = images->width*images->height;

    for (int imageIdx = 0; imageIdx < images->count; imageIdx++) {
        const int imageOffset = imageIdx * imageSize;
        float* imageD = images->devData + imageOffset;
        int* imageValidD = imagesValid->devData + imageOffset;
        double sum = 0;
        int count = 0;
        for (int i = 0; i < imageSize; i++) {
            sum += imageD[i] * imageD[i];
            count += imageValidD[i];
        }
        imagesSum->devData[imageIdx] = sum;
        imagesValidCount->devData[imageIdx] = count;
    }
}

} // namespace
