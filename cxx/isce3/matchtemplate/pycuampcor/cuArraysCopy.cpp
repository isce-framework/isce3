/**
 * @file cuArraysCopy.cu
 * @brief Utilities for copying/converting images to different format
 *
 * All methods are declared in cuAmpcorUtil.h
 * cudaArraysCopyToBatch to extract a batch of windows from the raw image
 *   various implementations include:
 *   1. fixed or varying offsets, as start pixels for windows
 *   2. complex to complex, usually
 *   3. complex to (amplitude,0), for TOPS
 *   4. real to complex, for real images
 * cuArraysCopyExtract to extract(shrink in size) from a batch of windows to another batch
 *   overloaded for different data types
 * cuArraysCopyInsert to insert a batch of windows (smaller in size) to another batch
 *   overloaded for different data types
 * cuArraysCopyPadded to insert a batch of windows to another batch while padding 0s for rest elements
 *   used for fft oversampling
 *   see also cuArraysPadding.cu for other zero-padding utilities
 * cuArraysAbs to convert between different data types
 */


// dependencies
#include "cuArrays.h"
#include "cudaUtil.h"
#include "float2.h"

namespace isce3::matchtemplate::pycuampcor {

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note used to extract chips from a raw secondary image with varying offsets
 * @param image1 Input image as a large chunk
 * @param lda1 the leading dimension of image1, usually, its width inNY
 * @param image2 Output images as a batch of chips
 * @param strideH (varying) offsets along height to extract chips
 * @param strideW (varying) offsets along width to extract chips
 */
void cuArraysCopyToBatchWithOffset(cuArrays<float2> *image1, const int lda1, cuArrays<float2> *image2,
    const int *offsetH, const int* offsetW)
{
    const int inNY = lda1;
    const float2* imageIn = image1->devData;
    float2* imageOut = image2->devData;
    const int* offsetX = offsetH;
    const int* offsetY = offsetW;
    const int outNX = image2->height;
    const int outNY = image2->width;

    for (int idxImage = 0; idxImage < image2->count; idxImage++) {
        for (int outx = 0; outx < outNX; outx++) {
            for (int outy = 0; outy < outNY; outy++) {
                int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
                int idxIn = (offsetX[idxImage]+outx)*inNY + offsetY[idxImage] + outy;
                imageOut[idxOut] = imageIn[idxIn];
            }
        }
    }
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note similar to cuArraysCopyToBatchWithOffset, but take amplitudes instead
 * @param image1 Input image as a large chunk
 * @param lda1 the leading dimension of image1, usually, its width inNY
 * @param image2 Output images as a batch of chips
 * @param strideH (varying) offsets along height to extract chips
 * @param strideW (varying) offsets along width to extract chips
 */
void cuArraysCopyToBatchAbsWithOffset(cuArrays<float2> *image1, const int lda1, cuArrays<float2> *image2,
    const int *offsetH, const int* offsetW)
{
    const int* offsetX = offsetH;
    const int* offsetY = offsetW;
    const int outNX = image2->height;
    const int outNY = image2->width;
    const float2* imageIn = image1->devData;
    const int inNY = lda1;
    float2* imageOut = image2->devData;
    for (int idxImage = 0; idxImage < image2->count; idxImage++) {
        for (int outx = 0; outx < outNX; outx++) {
            for (int outy = 0; outy < outNY; outy++) {
                int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
                int idxIn = (offsetX[idxImage]+outx)*inNY + offsetY[idxImage] + outy;
                imageOut[idxOut] = make_float2(complexAbs(imageIn[idxIn]), 0.0);
            }
        }
    }
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note used to load real images
 * @param image1 Input image as a large chunk
 * @param lda1 the leading dimension of image1, usually, its width inNY
 * @param image2 Output images as a batch of chips
 * @param strideH (varying) offsets along height to extract chips
 * @param strideW (varying) offsets along width to extract chips
 */
void cuArraysCopyToBatchWithOffsetR2C(cuArrays<float> *image1, const int lda1, cuArrays<float2> *image2,
    const int *offsetH, const int* offsetW)
{
    throw std::runtime_error("cuArraysCopyToBatchWithOffsetR2C_kernel not implemented");
}

/**
 * Copy a tile of images to another image, with starting pixels offsets
 * @param[in] imageIn input images of dimension nImages*inNX*inNY
 * @param[out] imageOut output images of dimension nImages*outNX*outNY
 * @param[in] offsets, varying offsets for extraction
 */
template<typename T>
void cuArraysCopyExtract(cuArrays<T> *imagesIn, cuArrays<T> *imagesOut, cuArrays<int2> *offsets)
{
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;
    const int outNX = imagesOut->height;
    const int outNY = imagesOut->width;
    for (int idxImage = 0; idxImage < imagesOut->count; idxImage++) {
        for (int outx = 0; outx < outNX; outx++) {
            for (int outy = 0; outy < outNY; outy++) {
                const int idxOut = (idxImage * outNX + outx)*outNY+outy;
                const int idxIn = (idxImage*inNX + outx + offsets->devData[idxImage].x)*inNY
                                                 + outy + offsets->devData[idxImage].y;
                imagesOut->devData[idxOut] = imagesIn->devData[idxIn];
            }
        }
    }
}

// instantiate the above template for the data types we need
template void cuArraysCopyExtract(cuArrays<float> *in, cuArrays<float> *out, cuArrays<int2> *offsets);
template void cuArraysCopyExtract(cuArrays<float2> *in, cuArrays<float2> *out, cuArrays<int2> *offsets);

// correlation surface extraction (Minyan Zhong)

/**
 * copy a tile of images to another image, with starting pixels offsets accouting for boundary
 * @param[in] imageIn inut images
 * @param[out] imageOut output images of dimension nImages*outNX*outNY
 */
void cuArraysCopyExtractCorr(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, cuArrays<int> *imagesValid, cuArrays<int2> *maxloc)
{
    const int outNX = imagesOut->height;
    const int outNY = imagesOut->width;
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;

    for (int imageIdx = 0; imageIdx < imagesOut->count; imageIdx++) {
        for (int outx = 0; outx < outNX; outx++) {
            for (int outy = 0; outy < outNY; outy++) {
                // Find the corresponding input.
                int inx = outx + maxloc->devData[imageIdx].x - outNX/2;
                int iny = outy + maxloc->devData[imageIdx].y - outNY/2;

                // Find the location in flattened array.
                int idxOut = (imageIdx * outNX + outx) * outNY + outy;
                int idxIn = (imageIdx * inNX + inx) * inNY + iny;

                // check whether inside of the input image
                if (inx>=0 && iny>=0 && inx<inNX && iny<inNY) {
                    // inside the boundary, copy over and mark the pixel as valid (1)
                    imagesOut->devData[idxOut] = imagesIn->devData[idxIn];
                    imagesValid->devData[idxOut] = 1;
                } else {
                    // outside, set it to 0 and mark the pixel as invalid (0)
                    imagesOut->devData[idxOut] = 0.0f;
                    imagesValid->devData[idxOut] = 0;
                }
            }
        }
    }
}

// end of correlation surface extraction (Minyan Zhong)


void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float> *imagesOut, int2 offset)
{
    //assert(imagesIn->height >= imagesOut && inNY >= outNY);
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;
    const int outNX = imagesOut->height;
    const int outNY = imagesOut->width;

    for (int imageIdx = 0; imageIdx < imagesOut->count; imageIdx++) {
        for (int outx = 0; outx < outNX; outx++) {
            for (int outy = 0; outy < outNY; outy++) {
                int idxOut = (imageIdx * outNX + outx) * outNY + outy;
                int idxIn = (imageIdx * inNX + outx + offset.x) * inNY + outy + offset.y;
                imagesOut->devData[idxOut] = imagesIn->devData[idxIn].x;
            }
        }
    }
}

/**
 * copy/extract images from a large size to
 * a smaller size from the location (offsetX, offsetY)
 */
template<typename T>
void cuArraysCopyExtract(cuArrays<T> *imagesIn, cuArrays<T> *imagesOut, int2 offset)
{
    //assert(imagesIn->height >= imagesOut && inNY >= outNY);
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;
    const int outNX = imagesOut->height;
    const int outNY = imagesOut->width;

    for (int imageIdx = 0; imageIdx < imagesOut->count; imageIdx++) {
        for (int outx = 0; outx < outNX; outx++) {
            for (int outy = 0; outy < outNY; outy++) {
                int idxOut = (imageIdx * outNX + outx) * outNY + outy;
                int idxIn = (imageIdx * inNX + outx + offset.x) * inNY + outy + offset.y;
                imagesOut->devData[idxOut] = imagesIn->devData[idxIn];
            }
        }
    }
}

// instantiate the above template for the data types we need
template void cuArraysCopyExtract(cuArrays<float> *in, cuArrays<float> *out, int2 offset);
template void cuArraysCopyExtract(cuArrays<float2> *in, cuArrays<float2> *out, int2 offset);
template void cuArraysCopyExtract(cuArrays<float3> *in, cuArrays<float3> *out, int2 offset);

/**
 * copy/insert images from a smaller size to a larger size from the location (offsetX, offsetY)
 */
template<typename T>
void cuArraysCopyInsert(cuArrays<T> *imageIn, cuArrays<T> *imageOut, int offsetX, int offsetY)
{
    const int inNX = imageIn->height;
    const int inNY = imageIn->width;
    const int outNY = imageOut->width;

    for (int inx = 0; inx < inNX; inx++) {
        for (int iny = 0; iny < inNY; iny++) {
            const int idxOut = IDX2R(inx + offsetX, iny + offsetY, outNY);
            const int idxIn = IDX2R(inx, iny, inNY);
            imageOut->devData[idxOut] = imageIn->devData[idxIn];
        }
    }
}

// instantiate the above template for the data types we need
template void cuArraysCopyInsert(cuArrays<float2>* in, cuArrays<float2>* out, int offX, int offY);
template void cuArraysCopyInsert(cuArrays<float3>* in, cuArrays<float3>* out, int offX, int offY);
template void cuArraysCopyInsert(cuArrays<float>* in, cuArrays<float>* out, int offX, int offY);
template void cuArraysCopyInsert(cuArrays<int>* in, cuArrays<int>* out, int offX, int offY);

/**
 * copy images from a smaller size to a larger size while padding 0 for extra elements
 */
template<typename T_in, typename T_out>
void cuArraysCopyPadded(cuArrays<T_in> *imageIn, cuArrays<T_out> *imageOut)
{
    for (int z = 0; z < imageIn->count; z++) {
        for (int i = 0; i < imageOut->height; i++) {
            for (int j = 0; j < imageOut->width; j++) {
                int idxOut = IDX2R(i, j, imageOut->width) + z * imageOut->size;
                if (i < imageIn->height && j < imageIn->width) {
                    int idxIn = IDX2R(i, j, imageIn->width) + z * imageIn->size;
                    imageOut->devData[idxOut] = T_out{imageIn->devData[idxIn]};
                } else {
                    imageOut->devData[idxOut] = T_out{0};
                }
            }
        }
    }
}

// instantiate the above template for the data types we need
template void cuArraysCopyPadded(cuArrays<float> *imageIn, cuArrays<float> *imageOut);
template void cuArraysCopyPadded(cuArrays<float> *imageIn, cuArrays<float2> *imageOut);
template void cuArraysCopyPadded(cuArrays<float2> *imageIn, cuArrays<float2> *imageOut);

/**
 * Obtain abs (amplitudes) of complex images
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysAbs(cuArrays<float2> *image1, cuArrays<float> *image2)
{
    int size = image1->getSize();
    for (int i = 0; i < size; i++)
        image2->devData[i] = complexAbs(image1->devData[i]);
}

} // namespace
