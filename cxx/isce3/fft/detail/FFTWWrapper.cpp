#include "FFTWWrapper.h"

#include <isce3/except/Error.h>

namespace isce { namespace fft { namespace detail {

static
void setNumThreadsf(int threads)
{
    // perform one-time initialization required to use threads
    static bool initialized(false);
    if (!initialized) {
        int status = fftwf_init_threads();
        if (status == 0) {
            throw isce::except::RuntimeError(ISCE_SRCINFO(), "multi-threaded FFTW initialization failed");
        }
        initialized = true;
    }

    // set max number of threads to use
    fftwf_plan_with_nthreads(threads);
}

static
void setNumThreads(int threads)
{
    // perform one-time initialization required to use threads
    static bool initialized(false);
    if (!initialized) {
        int status = fftw_init_threads();
        if (status == 0) {
            throw isce::except::RuntimeError(ISCE_SRCINFO(), "multi-threaded FFTW initialization failed");
        }
        initialized = true;
    }

    // set max number of threads to use
    fftw_plan_with_nthreads(threads);
}

fftwf_plan
initPlan(int rank, const int * n, int howmany,
         std::complex<float> * in,
         const int * inembed, int istride, int idist,
         std::complex<float> * out,
         const int * onembed, int ostride, int odist,
         int sign, unsigned flags, int threads)
{
    setNumThreadsf(threads);

    return fftwf_plan_many_dft(
            rank, n, howmany,
            reinterpret_cast<fftwf_complex *>(in),
            inembed, istride, idist,
            reinterpret_cast<fftwf_complex *>(out),
            onembed, ostride, odist,
            sign, flags);
}

fftw_plan
initPlan(int rank, const int * n, int howmany,
         std::complex<double> * in,
         const int * inembed, int istride, int idist,
         std::complex<double> * out,
         const int * onembed, int ostride, int odist,
         int sign, unsigned flags, int threads)
{
    setNumThreads(threads);

    return fftw_plan_many_dft(
            rank, n, howmany,
            reinterpret_cast<fftw_complex *>(in),
            inembed, istride, idist,
            reinterpret_cast<fftw_complex *>(out),
            onembed, ostride, odist,
            sign, flags);
}

fftwf_plan
initPlan(int rank, const int * n, int howmany,
         float * in,
         const int * inembed, int istride, int idist,
         std::complex<float> * out,
         const int * onembed, int ostride, int odist,
         int, unsigned flags, int threads)
{
    setNumThreadsf(threads);

    return fftwf_plan_many_dft_r2c(
            rank, n, howmany,
            in,
            inembed, istride, idist,
            reinterpret_cast<fftwf_complex *>(out),
            onembed, ostride, odist,
            flags);
}

fftw_plan
initPlan(int rank, const int * n, int howmany,
         double * in,
         const int * inembed, int istride, int idist,
         std::complex<double> * out,
         const int * onembed, int ostride, int odist,
         int, unsigned flags, int threads)
{
    setNumThreads(threads);

    return fftw_plan_many_dft_r2c(
            rank, n, howmany,
            in,
            inembed, istride, idist,
            reinterpret_cast<fftw_complex *>(out),
            onembed, ostride, odist,
            flags);
}

fftwf_plan
initPlan(int rank, const int * n, int howmany,
         std::complex<float> * in,
         const int * inembed, int istride, int idist,
         float * out,
         const int * onembed, int ostride, int odist,
         int, unsigned flags, int threads)
{
    setNumThreadsf(threads);

    return fftwf_plan_many_dft_c2r(
            rank, n, howmany,
            reinterpret_cast<fftwf_complex *>(in),
            inembed, istride, idist,
            out,
            onembed, ostride, odist,
            flags);
}

fftw_plan
initPlan(int rank, const int * n, int howmany,
         std::complex<double> * in,
         const int * inembed, int istride, int idist,
         double * out,
         const int * onembed, int ostride, int odist,
         int, unsigned flags, int threads)
{
    setNumThreads(threads);

    return fftw_plan_many_dft_c2r(
            rank, n, howmany,
            reinterpret_cast<fftw_complex *>(in),
            inembed, istride, idist,
            out,
            onembed, ostride, odist,
            flags);
}

void executePlan(const fftwf_plan plan)
{
    return fftwf_execute(plan);
}

void executePlan(const fftw_plan plan)
{
    return fftw_execute(plan);
}

void destroyPlan(fftwf_plan plan)
{
    if (plan) {
        fftwf_destroy_plan(plan);
    }
}

void destroyPlan(fftw_plan plan)
{
    if (plan) {
        fftw_destroy_plan(plan);
    }
}

}}}
