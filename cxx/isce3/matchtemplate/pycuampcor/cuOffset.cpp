/*
 * @file cuOffset.cu
 * @brief Utilities used to determine the offset field
 *
 */

// my module dependencies
#include "cuAmpcorUtil.h"

// for FLT_MAX
#include <cfloat>
#include <limits>
#include "float2.h"

namespace isce3::matchtemplate::pycuampcor {

// kernel for 2D array(image), find max value and location
void  cudaKernel_maxloc2D(const float* const images, int2* maxloc, float* maxval,
    const size_t imageNX, const size_t imageNY, const size_t nImages)
{
    const int imageSize = imageNX * imageNY;

    for (int bid = 0; bid < nImages; bid++) {
        float my_maxval = std::numeric_limits<float>::min();
        int2 my_maxloc;
        const float* image = &images[bid * imageSize];
        for (int i = 0; i < imageSize; i++) {
            if (image[i] > my_maxval) {
                my_maxval = image[i];
                my_maxloc = make_int2(i / imageNY, i % imageNY);
            }
        }
        maxval[bid] = my_maxval;
        maxloc[bid] = my_maxloc;
    }
}

/**
 * Find both the maximum value and the location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[out] maxval arrays to hold the max values
 * @param[out] maxloc arrays to hold the max locations
 * @note This routine is overloaded with the routine without maxval
 */
void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc,
                      cuArrays<float> *maxval)
{
    cudaKernel_maxloc2D(images->devData, maxloc->devData, maxval->devData,
            images->height, images->width, images->count);
}

/**
 * Determine the final offset value
 * @param[in] offsetInit max location (adjusted to the starting location for extraction) determined from
 *   the cross-correlation before oversampling, in dimensions of pixel
 * @param[in] offsetZoomIn max location from the oversampled cross-correlation surface
 * @param[out] offsetFinal the combined offset value
 * @param[in] OversampleRatioZoomIn the correlation surface oversampling factor
 * @param[in] OversampleRatioRaw the oversampling factor of reference/secondary windows before cross-correlation
 * @param[in] xHalfRangInit the original half search range along x, to be subtracted
 * @param[in] yHalfRangInit the original half search range along y, to be subtracted
 *
 * 1. Cross-correlation is performed at first for the un-oversampled data with a larger search range.
 *   The secondary window is then extracted to a smaller size (a smaller search range) around the max location.
 *   The extraction starting location (offsetInit) - original half search range (xHalfRangeInit, yHalfRangeInit)
 *        = pixel size offset
 * 2. Reference/secondary windows are then oversampled by OversampleRatioRaw, and cross-correlated.
 * 3. The correlation surface is further oversampled by OversampleRatioZoomIn.
 *    The overall oversampling factor is OversampleRatioZoomIn*OversampleRatioRaw.
 *    The max location in oversampled correlation surface (offsetZoomIn) / overall oversampling factor
 *        = subpixel offset
 *    Final offset =  pixel size offset +  subpixel offset
 */
void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetZoomIn,
    cuArrays<float2> *offsetFinal,
    int OverSampleRatioZoomin, int OverSampleRatioRaw,
    int xHalfRangeInit,  int yHalfRangeInit)
{
    int size = offsetInit->getSize();
    float OSratio = 1.0f/(float)(OverSampleRatioZoomin*OverSampleRatioRaw);
    float xoffset = xHalfRangeInit ;
    float yoffset = yHalfRangeInit ;

    float2* final = offsetFinal->devData;
    const int2* zoomin = offsetZoomIn->devData;
    const int2* init = offsetInit->devData;

    for (int idx = 0; idx < size; idx++) {
        final[idx].x = OSratio*(zoomin[idx].x ) + init[idx].x  - xoffset;
        final[idx].y = OSratio*(zoomin[idx].y ) + init[idx].y - yoffset;
    }
}

// function to compute the shift of center
static inline int2 adjustOffset(
    const int oldRange, const int newRange, const int maxloc)
{
    // determine the starting point around the maxloc
    // oldRange is the half search window size, e.g., = 32
    // newRange is the half extract size, e.g., = 4
    // maxloc is in range [0, 64]
    // we want to extract \pm 4 centered at maxloc
    // Examples:
    // 1. maxloc = 40: we set start=maxloc-newRange=36, and extract [36,44), shift=0
    // 2. maxloc = 2, start=-2: we set start=0, shift=-2,
    //   (shift means the max is -2 from the extracted center 4)
    // 3. maxloc =64, start=60: set start=56, shift = 4
    //   (shift means the max is 4 from the extracted center 60).

    // shift the max location by -newRange to find the start
    int start = maxloc - newRange;
    // if start is within the range, the max location will be in the center
    int shift = 0;
    // right boundary
    int rbound = 2*(oldRange-newRange);
    if(start<0)     // if exceeding the limit on the left
    {
        // set start at 0 and record the shift of center
        shift = -start;
        start = 0;
    }
    else if(start > rbound ) // if exceeding the limit on the right
    {
        //
        shift = start-rbound;
        start = rbound;
    }
    return make_int2(start, shift);
}

// kernel for cuDetermineSecondaryExtractOffset
void cudaKernel_determineSecondaryExtractOffset(int2 * maxLoc, int2 *shift,
    const int imageIndex, int xOldRange, int yOldRange, int xNewRange, int yNewRange)
{
    // get the starting pixel (stored back to maxloc) and shift
    int2 result = adjustOffset(xOldRange, xNewRange, maxLoc[imageIndex].x);
    maxLoc[imageIndex].x = result.x;
    shift[imageIndex].x = result.y;
    result = adjustOffset(yOldRange, yNewRange, maxLoc[imageIndex].y);
    maxLoc[imageIndex].y = result.x;
    shift[imageIndex].y = result.y;
}

/**
 * Determine the secondary window extract offset from the max location
 * @param[in] xOldRange, yOldRange are (half) search ranges in first step
 * @param[in] xNewRange, yNewRange are (half) search range
 *
 * After the first run of cross-correlation, with a larger search range,
 *  We now choose a smaller search range around the max location for oversampling.
 *  This procedure is used to determine the starting pixel locations for extraction.
 */
void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, cuArrays<int2> *maxLocShift,
    int xOldRange, int yOldRange, int xNewRange, int yNewRange)
{
    for (int i = 0; i < maxLoc->size; i++)
        cudaKernel_determineSecondaryExtractOffset(
                maxLoc->devData, maxLocShift->devData,
                i, xOldRange, yOldRange, xNewRange, yNewRange);
}

} // namespace
