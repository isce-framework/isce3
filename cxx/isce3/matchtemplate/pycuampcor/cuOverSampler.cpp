/* 
 * @file cuOverSampler.cu
 * @brief Implementations of cuOverSamplerR2R (C2C) class
 */

// my declarations
#include "cuOverSampler.h"

// dependencies
#include "cuArrays.h"
#include "cuArrays.h"
#include "cuAmpcorUtil.h"

namespace isce3::matchtemplate::pycuampcor {

/**
 * Constructor for cuOversamplerC2C
 * @param input image size inNX x inNY
 * @param output image size outNX x outNY
 * @param nImages batches
 */
cuOverSamplerC2C::cuOverSamplerC2C(
        cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut,
        int inNX, int inNY, int outNX, int outNY, int nImages)
{
    
    int inNXp2 = inNX;
    int inNYp2 = inNY;
    int outNXp2 = outNX;
    int outNYp2 = outNY;
    
    /* if expanded to 2^n
    int inNXp2 = nextpower2(inNX);
    int inNYp2 = nextpower2(inNY);
    int outNXp2 = inNXp2*outNX/inNX;
    int outNYp2 = inNYp2*outNY/inNY; 
    */

    // set up work arrays
    workIn = new cuArrays<float2>(inNXp2, inNYp2, nImages);
    workIn->allocate();
    workOut = new cuArrays<float2>(outNXp2, outNYp2, nImages);
    workOut->allocate();

    // set up fft plans
    int imageSize = inNXp2*inNYp2;
    int n[NRANK] ={inNXp2, inNYp2};
    int fImageSize = inNXp2*inNYp2;
    int nOverSample[NRANK] = {outNXp2, outNYp2};
    int fImageOverSampleSize = outNXp2*outNYp2;
    {
        fftwf_complex* in = (fftwf_complex*) imagesIn->devData;
        fftwf_complex* out = (fftwf_complex*) workIn->devData;
        forwardPlan = fftwf_plan_many_dft(NRANK, n, nImages,
                in, NULL, 1, imageSize,
                out, NULL, 1, fImageSize,
                FFTW_BACKWARD, FFTW_MEASURE);
    }
    {
        fftwf_complex* in = (fftwf_complex*) workOut->devData;
        fftwf_complex* out = (fftwf_complex*) imagesOut->devData;
        backwardPlan = fftwf_plan_many_dft(NRANK, nOverSample, nImages,
                in, NULL, 1, fImageOverSampleSize,
                out, NULL, 1, fImageOverSampleSize,
                FFTW_FORWARD, FFTW_MEASURE);
    }
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 * @param[in] method phase deramping method
 */
void cuOverSamplerC2C::execute(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, int method)
{   
    cuDeramp(method, imagesIn);
    fftwf_execute(forwardPlan);
    cuArraysPaddingMany(workIn, workOut);
    fftwf_execute(backwardPlan);
}

/// destructor
cuOverSamplerC2C::~cuOverSamplerC2C() 
{
    // destroy fft handles
    fftwf_destroy_plan(forwardPlan);
    fftwf_destroy_plan(backwardPlan);
    // deallocate work arrays
    delete(workIn);
    delete(workOut);	
}

// end of cuOverSamplerC2C

/**
 * Constructor for cuOversamplerR2R
 * @param input image size inNX x inNY
 * @param output image size outNX x outNY
 * @param nImages the number of images
 */
cuOverSamplerR2R::cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages)
{
    
    int inNXp2 = inNX;
    int inNYp2 = inNY;
    int outNXp2 = outNX;
    int outNYp2 = outNY;

    /* if expanded to 2^n
    int inNXp2 = nextpower2(inNX);
    int inNYp2 = nextpower2(inNY);
    int outNXp2 = inNXp2*outNX/inNX;
    int outNYp2 = inNYp2*outNY/inNY;
    */

    int imageSize = inNXp2 *inNYp2;
    int n[NRANK] ={inNXp2, inNYp2};
    int fImageSize = inNXp2*inNYp2;
    int nUpSample[NRANK] = {outNXp2, outNYp2};
    int fImageUpSampleSize = outNXp2*outNYp2;
    workSizeIn = new cuArrays<float2>(inNXp2, inNYp2, nImages);
    workSizeIn->allocate();
    workSizeOut = new cuArrays<float2>(outNXp2, outNYp2, nImages);
    workSizeOut->allocate();

    {
        fftwf_complex* in = (fftwf_complex*) workSizeIn->devData;
        forwardPlan = fftwf_plan_many_dft(NRANK, n, nImages,
                in, NULL, 1, imageSize,
                in, NULL, 1, fImageSize,
                FFTW_BACKWARD, FFTW_MEASURE);
    }
    {
        fftwf_complex* out = (fftwf_complex*) workSizeOut->devData;
        backwardPlan = fftwf_plan_many_dft(NRANK, nUpSample, nImages,
                out, NULL, 1, fImageUpSampleSize,
                out, NULL, 1, outNX*outNY,
                FFTW_FORWARD, FFTW_MEASURE);
    }
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 */
void cuOverSamplerR2R::execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut)
{
    cuArraysCopyPadded(imagesIn, workSizeIn);
    fftwf_execute(forwardPlan);
    cuArraysPaddingMany(workSizeIn, workSizeOut);
    fftwf_execute(backwardPlan);
    cuArraysCopyExtract(workSizeOut, imagesOut, make_int2(0,0));	
}

/// destructor
cuOverSamplerR2R::~cuOverSamplerR2R() 
{
    fftwf_destroy_plan(forwardPlan);
    fftwf_destroy_plan(backwardPlan);
    workSizeIn->deallocate();
    workSizeOut->deallocate();
}

} // namespace
