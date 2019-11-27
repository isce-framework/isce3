#pragma once

#include <complex>
#include <memory>
#include <type_traits>

#include "FFTWWrapper.h"

namespace isce { namespace fft { namespace detail {

template<int Sign, typename T>
class FFTPlanBase {
public:

    static_assert( Sign == FFTW_FORWARD || Sign == FFTW_BACKWARD );
    static_assert( std::is_same<T, float>::value || std::is_same<T, double>::value );

    FFTPlanBase();

    FFTPlanBase(std::complex<T> * out,
                std::complex<T> * in,
                int n,
                int batch = 1,
                unsigned flags = FFTW_MEASURE);

    template<int Rank>
    FFTPlanBase(std::complex<T> * out,
                std::complex<T> * in,
                const int (&n)[Rank],
                int batch = 1,
                unsigned flags = FFTW_MEASURE);

    FFTPlanBase(std::complex<T> * out,
                std::complex<T> * in,
                int n,
                int nembed,
                int stride,
                int dist,
                int batch = 1,
                unsigned flags = FFTW_MEASURE);

    template<int Rank>
    FFTPlanBase(std::complex<T> * out,
                std::complex<T> * in,
                const int (&n)[Rank],
                const int (&nembed)[Rank],
                int stride,
                int dist,
                int batch = 1,
                unsigned flags = FFTW_MEASURE);

    FFTPlanBase(std::complex<T> * out,
                std::complex<T> * in,
                int n,
                int inembed,
                int istride,
                int idist,
                int onembed,
                int ostride,
                int odist,
                int batch = 1,
                unsigned flags = FFTW_MEASURE);

    template<int Rank>
    FFTPlanBase(std::complex<T> * out,
                std::complex<T> * in,
                const int (&n)[Rank],
                const int (&inembed)[Rank],
                int istride,
                int idist,
                const int (&onembed)[Rank],
                int ostride,
                int odist,
                int batch = 1,
                unsigned flags = FFTW_MEASURE);

    explicit operator bool() const { return *_plan; }

    void execute() const;

protected:

    using fftw_plan_t = typename FFTWPlanType<T>::plan_t;

    template<typename U, typename V>
    FFTPlanBase(U * out,
                V * in,
                const int * n,
                const int * inembed,
                int istride,
                int idist,
                const int * onembed,
                int ostride,
                int odist,
                int batch,
                unsigned flags,
                int rank,
                int sign);

    std::shared_ptr<fftw_plan_t> _plan;
};

template<int N>
int product(const int (&arr)[N]);

}}}

#define ISCE_FFT_DETAIL_FFTPLANBASE_ICC
#include "FFTPlanBase.icc"
#undef ISCE_FFT_DETAIL_FFTPLANBASE_ICC
