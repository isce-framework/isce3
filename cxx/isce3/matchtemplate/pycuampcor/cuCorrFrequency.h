/*
 * @file  cuCorrFrequency.h
 * @brief A class performs cross correlation in frequency domain
 */

// code guard
#ifndef __CUCORRFREQUENCY_H
#define __CUCORRFREQUENCY_H

// dependencies
#include "cudaUtil.h"
#include "cuArrays.h"

#include <fftw3.h>

namespace isce3::matchtemplate::pycuampcor {

class cuFreqCorrelator
{
private:
    // handles for forward/backward fft
    fftwf_plan forwardPlan1;
    fftwf_plan forwardPlan2;
    fftwf_plan backwardPlan;
    // work data
    cuArrays<float2> *workFM;
    cuArrays<float2> *workFS;
    cuArrays<float> *workT;

public:
    // constructor
    cuFreqCorrelator(cuArrays<float>* images, int imageNX, int imageNY, int nImages);
    // destructor
    ~cuFreqCorrelator();
    // executor
    void execute(cuArrays<float> *templates, cuArrays<float> *images, cuArrays<float> *results);
};

} // namespace

#endif //__CUCORRFREQUENCY_H
// end of file
