/*
 * @file cuCorrNormalizationSAT.cu
 * @brief Utilities to normalize the 2D correlation surface with the sum area table
 *
 */

// my declarations
#include "cuAmpcorUtil.h"
// for FLT_EPSILON
#include <float.h>
#include <math.h>


namespace isce3::matchtemplate::pycuampcor {

/**
 * kernel for sum value^2 (std)
 * compute the sum value square (std) of the reference image
 * @param[out] sum2 sum of value square
 * @param[in] images the reference images
 * @param[in] n total elements in one image nx*ny
 * @param[in] batch number of images
 **/

void sum_square_kernel(float *sum2, const float *images, int n, int batch)
{
    for (int j = 0; j < batch; j++) {
        const float* image = &images[j * n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += image[i] * image[i];
        }
        sum2[j] = sum;
    }
}

/**
 * kernel for 2d sum area table
 * Compute the (inclusive) sum area table of the value and value^2 of a batch of 2d images.
 * @param[out] sat the sum area table
 * @param[out] sat2 the sum area table of value^2
 * @param[in] data search images
 * @param[in] nx image height (subleading dimension)
 * @param[in] ny image width (leading dimension)
 * @param[in] batch number of images
 **/

void sat2d_kernel(float *sat, float * sat2, const float *data, int nx, int ny, int batch, int imagecount)
{
    for (int imageid = 0; imageid < imagecount; imageid++) {
        for (int tid = 0; tid < 256; tid++) {

            // compute prefix-sum along row at first
            // the number of rows may be bigger than the number of threads, iterate
            for (int row = tid; row < nx; row += 256) {
                // running sum for value and value^2
                float sum = 0.0f;
                float sum2 = 0.0f;
                // starting position for this row
                int index = (imageid*nx+row)*ny;
                // iterative over column
                for (int i=0; i<ny; i++, index++) {
                    float val = data[index];
                    sum += val;
                    sat[index] = sum;
                    sum2 += val*val;
                    sat2[index] = sum2;
                }
            }
        }
    }

    for (int imageid = 0; imageid < imagecount; imageid++) {
        for (int tid = 0; tid < 256; tid++) {
            // compute prefix-sum along column
            for (int col = tid; col < ny; col += 256) {

                // start position of the current column
                int index = col + imageid*nx*ny;

                // assign sum with the first line value
                float sum = sat[index];
                float sum2 = sat2[index];
                // iterative over rest lines
                for (int i=1; i<nx; i++) {
                    index += ny;
                    sum += sat[index];
                    sat[index] = sum;
                    sum2 += sat2[index];
                    sat2[index] = sum2;
                }
            }
        }
    }
}

void cuCorrNormalizeSAT_kernel(float *correlation, const float *referenceSum2, const float *secondarySat,
    const float *secondarySat2, const int corNX, const int corNY, const int referenceNX, const int referenceNY,
    const int secondaryNX, const int secondaryNY, const int imagecount)
{
    for (int imageid = 0; imageid < imagecount; imageid++) {
        for (int tx = 0; tx < corNX; tx++) {
            for (int ty = 0; ty < corNY; ty++) {

                // get the reference std
                float refSum2 = referenceSum2[imageid];

                // compute the sum and sum square of the search image from the sum area table
                // sum
                const float *sat = secondarySat + imageid*secondaryNX*secondaryNY;
                // get sat values for four corners
                float topleft = (tx > 0 && ty > 0) ? sat[(tx-1)*secondaryNY+(ty-1)] : 0.0;
                float topright = (tx > 0 ) ? sat[(tx-1)*secondaryNY+(ty+referenceNY-1)] : 0.0;
                float bottomleft = (ty > 0) ? sat[(tx+referenceNX-1)*secondaryNY+(ty-1)] : 0.0;
                float bottomright = sat[(tx+referenceNX-1)*secondaryNY+(ty+referenceNY-1)];
                // get the sum
                float secondarySum = bottomright + topleft - topright - bottomleft;
                // sum of value^2
                const float *sat2 = secondarySat2 + imageid*secondaryNX*secondaryNY;
                // get sat2 values for four corners
                topleft = (tx > 0 && ty > 0) ? sat2[(tx-1)*secondaryNY+(ty-1)] : 0.0;
                topright = (tx > 0 ) ? sat2[(tx-1)*secondaryNY+(ty+referenceNY-1)] : 0.0;
                bottomleft = (ty > 0) ? sat2[(tx+referenceNX-1)*secondaryNY+(ty-1)] : 0.0;
                bottomright = sat2[(tx+referenceNX-1)*secondaryNY+(ty+referenceNY-1)];
                float secondarySum2 = bottomright + topleft - topright - bottomleft;

                // compute the normalization
                float norm2 = (secondarySum2-secondarySum*secondarySum/(referenceNX*referenceNY))*refSum2;
                // normalize the correlation surface
                correlation[(imageid*corNX+tx)*corNY+ty] *= 1 / sqrtf(norm2 + FLT_EPSILON);
            }
        }
    }
}


void cuCorrNormalizeSAT(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary,
    cuArrays<float> * referenceSum2, cuArrays<float> *secondarySat, cuArrays<float> *secondarySat2)
{
    // compute the std of reference image
    // note that the mean is already subtracted
    sum_square_kernel(referenceSum2->devData, reference->devData,
            reference->width * reference->height, reference->count);

    // compute the sum area table of the search images
    sat2d_kernel(secondarySat->devData, secondarySat2->devData, secondary->devData,
        secondary->height, secondary->width, secondary->count, reference->count);

    cuCorrNormalizeSAT_kernel(correlation->devData,
        referenceSum2->devData, secondarySat->devData, secondarySat2->devData,
        correlation->height, correlation->width,
        reference->height, reference->width,
        secondary->height, secondary->width, correlation->count);
}

} // namespace
