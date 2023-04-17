/*
 * @file cuAmpcorUtil.h
 * @brief Header file to include various routines for cuAmpcor
 *
 *
 */

// code guard
#ifndef __CUAMPCORUTIL_H
#define __CUAMPCORUTIL_H

#include "cuArrays.h"
#include "cuAmpcorParameter.h"


namespace isce3::matchtemplate::pycuampcor {

//in cuArraysCopy.cu: various utilities for copy images file in gpu memory
void cuArraysCopyToBatchWithOffset(cuArrays<float2> *image1, const int lda1, cuArrays<float2> *image2,
    const int *offsetH, const int* offsetW);
void cuArraysCopyToBatchAbsWithOffset(cuArrays<float2> *image1, const int lda1, cuArrays<float2> *image2,
    const int *offsetH, const int* offsetW);
void cuArraysCopyToBatchWithOffsetR2C(cuArrays<float> *image1, const int lda1, cuArrays<float2> *image2,
    const int *offsetH, const int* offsetW);

// same routine name overloaded for different data type
// extract data from a large image
template<typename T>
void cuArraysCopyExtract(cuArrays<T> *imagesIn, cuArrays<T> *imagesOut, cuArrays<int2> *offset);

void cuArraysCopyExtract(cuArrays<float2> *imagesIn, cuArrays<float> *imagesOut, int2 offset);

template<typename T>
void cuArraysCopyExtract(cuArrays<T> *imagesIn, cuArrays<T> *imagesOut, int2 offset);

template<typename T>
void cuArraysCopyInsert(cuArrays<T> *in, cuArrays<T> *out, int offsetX, int offsetY);

template<typename T_in, typename T_out>
void cuArraysCopyPadded(cuArrays<T_in> *imageIn, cuArrays<T_out> *imageOut);

void cuArraysAbs(cuArrays<float2> *image1, cuArrays<float> *image2);

// cuDeramp.cu: deramping phase
void cuDeramp(int method, cuArrays<float2> *images);
void cuDerampMethod1(cuArrays<float2> *images);

// cuArraysPadding.cu: various utilities for oversampling padding
void cuArraysPaddingMany(cuArrays<float2> *image1, cuArrays<float2> *image2);

//in cuCorrNormalization.cu: utilities to normalize the cross correlation function
void cuArraysSubtractMean(cuArrays<float> *images);
void cuCorrNormalize(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results);

// in cuCorrNormalizationSAT.cu: to normalize the cross correlation function with sum area table
void cuCorrNormalizeSAT(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary,
    cuArrays<float> * referenceSum2, cuArrays<float> *secondarySAT, cuArrays<float> *secondarySAT2);

//in cuOffset.cu: utitilies for determining the max locaiton of cross correlations or the offset
void cuArraysMaxloc2D(cuArrays<float> *images, cuArrays<int2> *maxloc, cuArrays<float> *maxval);
void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetZoomIn, cuArrays<float2> *offsetFinal,
                      int OverSampleRatioZoomin, int OverSampleRatioRaw,
                      int xHalfRangeInit,  int yHalfRangeInit);

void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, cuArrays<int2> *maxLocShift,
        int xOldRange, int yOldRange, int xNewRange, int yNewRange);

//in cuCorrTimeDomain.cu: cross correlation in time domain
void cuCorrTimeDomain(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results);

// For SNR estimation on Correlation surface (Minyan Zhong)
// implemented in cuArraysCopy.cu
void cuArraysCopyExtractCorr(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, cuArrays<int> *imagesValid, cuArrays<int2> *maxloc);
// implemented in cuCorrNormalization.cu
void cuArraysSumCorr(cuArrays<float> *images, cuArrays<int> *imagesValid, cuArrays<float> *imagesSum, cuArrays<int> *imagesValidCount);

// implemented in cuEstimateStats.cu
void cuEstimateSnr(cuArrays<float> *corrSum, cuArrays<int> *corrValidCount, cuArrays<float> *maxval, cuArrays<float> *snrValue);

// implemented in cuEstimateStats.cu
void cuEstimateVariance(cuArrays<float> *corrBatchRaw, cuArrays<int2> *maxloc, cuArrays<float> *maxval, int templateSize, cuArrays<float3> *covValue);

} // namespace

#endif

// end of file
