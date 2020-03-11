#pragma once

#include <complex>
#include <fftw3.h>

namespace isce { namespace fft { namespace detail {

template<typename T> struct FFTWPlanType {};
template<>           struct FFTWPlanType<float>  { using plan_t = fftwf_plan; };
template<>           struct FFTWPlanType<double> { using plan_t = fftw_plan; };

fftwf_plan
initPlan(int rank, const int * n, int howmany,
         std::complex<float> * in,
         const int * inembed, int istride, int idist,
         std::complex<float> * out,
         const int * onembed, int ostride, int odist,
         int sign, unsigned flags, int threads);

fftw_plan
initPlan(int rank, const int * n, int howmany,
         std::complex<double> * in,
         const int * inembed, int istride, int idist,
         std::complex<double> * out,
         const int * onembed, int ostride, int odist,
         int sign, unsigned flags, int threads);

fftwf_plan
initPlan(int rank, const int * n, int howmany,
         float * in,
         const int * inembed, int istride, int idist,
         std::complex<float> * out,
         const int * onembed, int ostride, int odist,
         int sign, unsigned flags, int threads);

fftw_plan
initPlan(int rank, const int * n, int howmany,
         double * in,
         const int * inembed, int istride, int idist,
         std::complex<double> * out,
         const int * onembed, int ostride, int odist,
         int sign, unsigned flags, int threads);

fftwf_plan
initPlan(int rank, const int * n, int howmany,
         std::complex<float> * in,
         const int * inembed, int istride, int idist,
         float * out,
         const int * onembed, int ostride, int odist,
         int sign, unsigned flags, int threads);

fftw_plan
initPlan(int rank, const int * n, int howmany,
         std::complex<double> * in,
         const int * inembed, int istride, int idist,
         double * out,
         const int * onembed, int ostride, int odist,
         int sign, unsigned flags, int threads);

void executePlan(const fftwf_plan);
void executePlan(const fftw_plan);

void destroyPlan(fftwf_plan);
void destroyPlan(fftw_plan);

}}}
