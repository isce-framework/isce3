/*
 * @file  cuDeramp.cpp
 * @brief Derampling a batch of 2D complex images
 *
 * A phase ramp is equivalent to a frequency shift in frequency domain,
 *   which needs to be removed (deramping) in order to move the band center
 *   to zero. This is necessary before oversampling a complex signal.
 * Method 1: each signal is decomposed into real and imaginary parts,
 *   and the average phase shift is obtained as atan(\sum imag / \sum real).
 *   The average is weighted by the amplitudes (coherence).
 * Method 0 or else: skip deramping
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

// kernel for linear deramping
static void cuLinearDeramp_kernel(float2 *images, const int imageNX, const int imageNY,
    const int imageSize, const int nImages, const float normCoef, const int axis)
{
    for (int k = 0; k < nImages; k++) {

        float2* image = images + k * imageSize;

        double phaseY = 0.0;
        if(axis != 0)
        {
            double2 phaseDiffY = make_double2(0.0, 0.0);
            for (int j = 0; j < imageNX; j++) {
                for (int i = 0; i < imageNY - 1; i++) {
                    const int pixelIdx = j * imageNY + i;
                    float2 cprod = complexMulConj(image[pixelIdx], image[pixelIdx+1]);
                    phaseDiffY += cprod;
                }
            }
            phaseY = atan2(phaseDiffY.y, phaseDiffY.x);
        }

        double phaseX = 0.0;
        if(axis != 1)
        {
            double2 phaseDiffX = make_double2(0.0, 0.0);
            for (int j = 0; j < imageNX - 1; j++) {
                for (int i = 0; i < imageNY; i++) {
                    const int pixelIdx = j * imageNY + i;
                    float2 cprod = complexMulConj(image[pixelIdx], image[pixelIdx+imageNY]);
                    phaseDiffX += cprod;
                }
            }
            phaseX = atan2(phaseDiffX.y, phaseDiffX.x);
        }

        for (int i = 0; i < imageSize; i++) {
            const int pixelIdxX = i / imageNY;
            const int pixelIdxY = i % imageNY;
            double phase = pixelIdxX*phaseX + pixelIdxY*phaseY;
            double phase_cos = cos(phase);
            double phase_sin = sin(phase);
            image[i] = make_float2(
                image[i].x*phase_cos - image[i].y*phase_sin,
                image[i].x*phase_sin + image[i].y*phase_cos);
        }
    }
}

/**
 * Deramp a complex signal with Method 1
 * @brief Each signal is decomposed into real and imaginary parts,
 *   and the average phase shift is obtained as atan(\sum imag / \sum real).
 * @param[inout] images input/output complex signals
 */
void cuLinearDeramp(cuArrays<float2> *images, const int axis)
{
    const int imageSize = images->width*images->height;
    const float invSize = 1.0f/imageSize;

    cuLinearDeramp_kernel(images->devData, images->height, images->width,
        imageSize, images->count, invSize, axis);
}

void cuDeramp(const int method, cuArrays<float2> *images, const int axis)
{
    switch(method) {
    case 1:
        cuLinearDeramp(images, axis);
        break;
    default:
        break;
    }
}

} // namespace
