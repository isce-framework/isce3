/*
 * @file cuOverSampler.h
 * @brief Oversampling with FFT padding method
 *
 * Define cuOverSampler class, to save fftw plans and perform oversampling calculations
 * For float images use cuOverSamplerR2R
 * For complex images use cuOverSamplerC2C
 * @todo use template class to unify these two classes
 */

#ifndef __CUOVERSAMPLER_H
#define __CUOVERSAMPLER_H

#include "cuArrays.h"
#include "cudaUtil.h"

#include <fftw3.h>

namespace isce3::matchtemplate::pycuampcor {

// FFT Oversampler for complex images
class cuOverSamplerC2C
{
private:
     fftwf_plan forwardPlan;   // forward fft handle
     fftwf_plan backwardPlan;  // backward fft handle
     cuArrays<float2> *workIn;  // work array to hold forward fft data
     cuArrays<float2> *workOut; // work array to hold padded data
public:
     // disable the default constructor
     cuOverSamplerC2C() = delete;
     // constructor
     cuOverSamplerC2C(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut,
             int inNX, int inNY, int outNX, int outNY, int nImages);
     // execute oversampling
     void execute(cuArrays<float2> *imagesIn, cuArrays<float2> *imagesOut, int deramp_method=0);
     // destructor
     ~cuOverSamplerC2C();
};

// FFT Oversampler for complex images
class cuOverSamplerR2R
{
private:
     fftwf_plan forwardPlan;
     fftwf_plan backwardPlan;
     cuArrays<float2> *workSizeIn;
     cuArrays<float2> *workSizeOut;

public:
    cuOverSamplerR2R() = delete;
    cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages);
    void execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut);
    ~cuOverSamplerR2R();
};

} // namespace

#endif //__CUOVERSAMPLER_H
// end of file



