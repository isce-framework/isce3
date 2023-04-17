/*
 * @file  cuArraysPadding.cu
 * @brief Utilities for padding zeros to cuArrays
 */

#include "cuAmpcorUtil.h"
#include "cudaUtil.h"
#include "float2.h"
#include <string.h>

namespace isce3::matchtemplate::pycuampcor {

inline float2 cmplxMul(float2 c, float a)
{
    return make_float2(c.x*a, c.y*a);
}

void cuArraysPaddingMany_kernel(
    const float2 *image1, const int height1, const int width1, const int size1,
    float2 *image2, const int height2, const int width2, const int size2, const float factor,
    int tx, int ty, int imgidx)
{
    if(tx < height1/2 && ty < width1/2)
    {

        int tx1 = height1 - 1 - tx;
        int ty1 = width1 -1 -ty;
        int tx2 = height2 -1 -tx;
        int ty2 = width2 -1 -ty;

        int stride1 = imgidx*size1;
        int stride2 = imgidx*size2;

        image2[IDX2R(tx,  ty,  width2)+stride2] = image1[IDX2R(tx,  ty,  width1)+stride1]*factor;
        image2[IDX2R(tx2, ty,  width2)+stride2] = cmplxMul(image1[IDX2R(tx1, ty,  width1)+stride1], factor);
        image2[IDX2R(tx,  ty2, width2)+stride2] = cmplxMul(image1[IDX2R(tx,  ty1, width1)+stride1], factor);
        image2[IDX2R(tx2, ty2, width2)+stride2] = cmplxMul(image1[IDX2R(tx1, ty1, width1)+stride1], factor);
    }
}

/**
 * Padding zeros for FFT oversampling
 * @param[in] image1 input images
 * @param[out] image2 output images
 * @note To keep the band center at (0,0), move quads to corners and pad zeros in the middle
 */
void cuArraysPaddingMany(cuArrays<float2> *image1, cuArrays<float2> *image2)
{
    memset(image2->devData, 0, image2->getByteSize());
    float factor = 1.0f/image1->size;

    for (int imgidx = 0; imgidx < image1->count; imgidx++) {
        for (int tx = 0; tx < image1->height / 2; tx++) {
            for (int ty = 0; ty < image1->width / 2; ty++) {
                cuArraysPaddingMany_kernel(
                    image1->devData, image1->height, image1->width, image1->size,
                    image2->devData, image2->height, image2->width, image2->size,
                    factor, tx, ty, imgidx);
            }
        }
    }
}

} // namespace
