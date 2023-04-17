/*
 * @file  cuCorrFrequency.cu
 * @brief A class performs cross correlation in frequency domain
 */

#include "cuCorrFrequency.h"
#include "cuAmpcorUtil.h"
#include "float2.h"


#include <fftw3.h>

namespace isce3::matchtemplate::pycuampcor {

/*
 * cuFreqCorrelator Constructor
 * @param imageNX height of each image
 * @param imageNY width of each image
 * @param nImages number of images in the batch
 */
cuFreqCorrelator::cuFreqCorrelator(
        cuArrays<float>* images,
        int imageNX, int imageNY, int nImages)
{

    int imageSize = imageNX*imageNY;
    int fImageSize = imageNX*(imageNY/2+1);
    int n[NRANK] ={imageNX, imageNY};

    // set up work arrays
    workFM = new cuArrays<float2>(imageNX, (imageNY/2+1), nImages);
    workFM->allocate();
    workFS = new cuArrays<float2>(imageNX, (imageNY/2+1), nImages);
    workFS->allocate();
    workT = new cuArrays<float> (imageNX, imageNY, nImages);
    workT->allocate();

    // set up fft plans
    {
        float* in = workT->devData;
        fftwf_complex* out = (fftwf_complex*) workFM->devData;
        forwardPlan1 = fftwf_plan_many_dft_r2c(NRANK, n, nImages,
                in, NULL, 1, imageSize,
                out, NULL, 1, fImageSize,
                FFTW_MEASURE);
    }

    {
        float* in = images->devData;
        fftwf_complex* out = (fftwf_complex*) workFS->devData;
        forwardPlan2 = fftwf_plan_many_dft_r2c(NRANK, n, nImages,
                in, NULL, 1, imageSize,
                out, NULL, 1, fImageSize,
                FFTW_MEASURE);
    }

    {
        fftwf_complex* in = (fftwf_complex*) workFM->devData;
        float* out = workT->devData;
        backwardPlan = fftwf_plan_many_dft_c2r(NRANK, n, nImages,
                in, NULL, 1, fImageSize,
                out, NULL, 1, imageSize,
                FFTW_MEASURE);
    }
}

/// destructor
cuFreqCorrelator::~cuFreqCorrelator()
{
    fftwf_destroy_plan(forwardPlan1);
    fftwf_destroy_plan(forwardPlan2);
    fftwf_destroy_plan(backwardPlan);
    workFM->deallocate();
    workFS->deallocate();
    workT->deallocate();
}

// a = a^* * b
inline __host__ __device__ float2 cuMulConj(float2 a, float2 b)
{
    return make_float2(a.x*b.x + a.y*b.y, -a.y*b.x + a.x*b.y);
}

/**
 * Execute the cross correlation
 * @param[in] templates the reference windows
 * @param[in] images the search windows
 * @param[out] results the correlation surfaces
 */

void cuFreqCorrelator::execute(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results)
{
    // pad the reference windows to the the size of search windows
    cuArraysCopyPadded(templates, workT);
    // forward fft to frequency domain
    fftwf_execute(forwardPlan1);
    fftwf_execute(forwardPlan2);
    // fftw doesn't normalize, so manually get the image size for normalization
    float coef = 1.0/(images->size);
    for (int i = 0; i < workFM->getSize(); i++) {
        workFM->devData[i] = cuMulConj(workFM->devData[i], workFS->devData[i]) * coef;
    }
    // backward fft to get correlation surface in time domain
    fftwf_execute(backwardPlan);
    // extract to get proper size of correlation surface
    cuArraysCopyExtract(workT, results, make_int2(0, 0));
    // all done
}

} // namespace
