/**
 * @file  cuEstimateStats.cu
 * @brief Estimate the statistics of the correlation surface
 *
 * 9/23/2017, Minyan Zhong
 */

#include "cuArrays.h"
#include "float2.h"
#include <cfloat>
#include "cudaUtil.h"
#include "cuAmpcorUtil.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

namespace isce3::matchtemplate::pycuampcor {

/**
 * Estimate the signal to noise ratio (SNR) of the correlation surface
 * @param[in] corrSum the sum of the correlation surface
 * @param[in] corrValidCount the number of valid pixels contributing to sum
 * @param[out] snrValue return snr value
 */
void cuEstimateSnr(cuArrays<float> *corrSum, cuArrays<int> *corrValidCount, cuArrays<float> *maxval, cuArrays<float> *snrValue)
{

    int size = corrSum->getSize();

    for (int i = 0; i < size; i++) {
        float maxvalsq = maxval->devData[i] * maxval->devData[i];
        float mean = (corrSum->devData[i] - maxvalsq) / (corrValidCount->devData[i] - 1);
        snrValue->devData[i] = maxvalsq / mean;
    }
}

// helper function for cuEstimateVariance
void cudaKernel_estimateVar(const float* corrBatchRaw, const int NX, const int NY, const int2* maxloc,
        const float* maxval, const int templateSize, float3* covValue, const int idxImage)
{
    // Preparation.
    int px = maxloc[idxImage].x;
    int py = maxloc[idxImage].y;
    float peak = maxval[idxImage];

    // Check if maxval is on the margin.
    if (px-1 < 0 || py-1 <0 || px + 1 >=NX || py+1 >=NY)  {

        covValue[idxImage] = make_float3(99.0, 99.0, 0.0);

    }
    else {
        int offset = NX * NY * idxImage;
        int idx00 = offset + (px - 1) * NY + py - 1;
        int idx01 = offset + (px - 1) * NY + py    ;
        int idx02 = offset + (px - 1) * NY + py + 1;
        int idx10 = offset + (px    ) * NY + py - 1;
        int idx11 = offset + (px    ) * NY + py    ;
        int idx12 = offset + (px    ) * NY + py + 1;
        int idx20 = offset + (px + 1) * NY + py - 1;
        int idx21 = offset + (px + 1) * NY + py    ;
        int idx22 = offset + (px + 1) * NY + py + 1;

        // second-order derivatives
        float dxx = - ( corrBatchRaw[idx21] + corrBatchRaw[idx01] - 2.0*corrBatchRaw[idx11] );
        float dyy = - ( corrBatchRaw[idx12] + corrBatchRaw[idx10] - 2.0*corrBatchRaw[idx11] ) ;
        float dxy = ( corrBatchRaw[idx22] + corrBatchRaw[idx00] - corrBatchRaw[idx20] - corrBatchRaw[idx02] ) *0.25;

        float n2 = fmaxf(1.0 - peak, 0.0);

        dxx = dxx * templateSize;
        dyy = dyy * templateSize;
        dxy = dxy * templateSize;

        float n4 = n2*n2;
        n2 = n2 * 2;
        n4 = n4 * 0.5 * templateSize;

        float u = dxy * dxy - dxx * dyy;
        float u2 = u*u;

        // if the Gaussian curvature is too small
        if (fabsf(u) < 1e-2) {
            covValue[idxImage] = make_float3(99.0, 99.0, 0.0);
        }
        else {
                float cov_xx = (- n2 * u * dyy + n4 * ( dyy*dyy + dxy*dxy) ) / u2;
                float cov_yy = (- n2 * u * dxx + n4 * ( dxx*dxx + dxy*dxy) ) / u2;
                float cov_xy = (  n2 * u * dxy - n4 * ( dxx + dyy ) * dxy ) / u2;
                covValue[idxImage] = make_float3(cov_xx, cov_yy, cov_xy);
        }
    }
}

/**
 * Estimate the variance of the correlation surface
 * @param[in] templateSize size of reference chip
 * @param[in] corrBatchRaw correlation surface
 * @param[in] maxloc maximum location
 * @param[in] maxval maximum value
 * @param[out] covValue variance value
 */
void cuEstimateVariance(cuArrays<float> *corrBatchRaw, cuArrays<int2> *maxloc, cuArrays<float> *maxval, int templateSize, cuArrays<float3> *covValue)
{
    int size = corrBatchRaw->count;
    // One dimensional launching parameters to loop over every correlation surface.
    for (int i = 0; i < size; i++)
        cudaKernel_estimateVar(corrBatchRaw->devData, corrBatchRaw->height, corrBatchRaw->width,
                maxloc->devData, maxval->devData, templateSize, covValue->devData, i);
}

} // namespace
