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

// kernel for cuDerampMethod1
static void cuDerampMethod1_kernel(float2 *images, const int imageNX, int const imageNY,
    const int imageSize, const int nImages, const float normCoef)
{
    for (int k = 0; k < nImages; k++) {

        float2* image = images + k * imageSize;

        double2 phaseDiffY = make_double2(0.0, 0.0);
        for (int j = 0; j < imageNX; j++) {
            for (int i = 0; i < imageNY - 1; i++) {
                const int pixelIdx = j * imageNY + i;
                float2 cprod = complexMulConj(image[pixelIdx], image[pixelIdx+1]);
                phaseDiffY += cprod;
            }
        }

        double2 phaseDiffX = make_double2(0.0, 0.0);
        for (int j = 0; j < imageNX - 1; j++) {
            for (int i = 0; i < imageNY; i++) {
                const int pixelIdx = j * imageNY + i;
                float2 cprod = complexMulConj(image[pixelIdx], image[pixelIdx+imageNY]);
                phaseDiffX += cprod;
            }
        }
        float phaseX = atan2f(phaseDiffX.y, phaseDiffX.x);
        float phaseY = atan2f(phaseDiffY.y, phaseDiffY.x);

        for (int i = 0; i < imageSize; i++) {
            const int pixelIdxX = i%imageNY;
            const int pixelIdxY = i/imageNY;
            float phase = pixelIdxX*phaseX + pixelIdxY*phaseY;
            float2 phase_factor = make_float2(cosf(phase), sinf(phase));
            image[i] *= phase_factor;
        }
    }
}

/**
 * Deramp a complex signal with Method 1
 * @brief Each signal is decomposed into real and imaginary parts,
 *   and the average phase shift is obtained as atan(\sum imag / \sum real).
 * @param[inout] images input/output complex signals
 */
void cuDerampMethod1(cuArrays<float2> *images)
{
    const int imageSize = images->width*images->height;
    const float invSize = 1.0f/imageSize;

    cuDerampMethod1_kernel(images->devData, images->height, images->width,
        imageSize, images->count, invSize);
}

void cuDeramp(int method, cuArrays<float2> *images)
{
    switch(method) {
    case 1:
        cuDerampMethod1(images);
        break;
    default:
        break;
    }
}

} // namespace
