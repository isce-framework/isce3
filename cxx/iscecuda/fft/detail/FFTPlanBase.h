#pragma once

#include <memory>
#include <thrust/complex.h>
#include <type_traits>

#include "CufftWrapper.h"

namespace isce { namespace cuda { namespace fft { namespace detail {

template<int Sign, typename T>
class FFTPlanBase {
public:

    static_assert( Sign == CUFFT_FORWARD || Sign == CUFFT_INVERSE, "" );
    static_assert( std::is_same<T, float>::value || std::is_same<T, double>::value, "" );

    FFTPlanBase();

    FFTPlanBase(thrust::complex<T> * out,
                thrust::complex<T> * in,
                int n,
                int batch = 1);

    template<int Rank>
    FFTPlanBase(thrust::complex<T> * out,
                thrust::complex<T> * in,
                const int (&n)[Rank],
                int batch = 1);

    FFTPlanBase(thrust::complex<T> * out,
                thrust::complex<T> * in,
                int n,
                int nembed,
                int stride,
                int dist,
                int batch = 1);

    template<int Rank>
    FFTPlanBase(thrust::complex<T> * out,
                thrust::complex<T> * in,
                const int (&n)[Rank],
                const int (&nembed)[Rank],
                int stride,
                int dist,
                int batch = 1);

    FFTPlanBase(thrust::complex<T> * out,
                thrust::complex<T> * in,
                int n,
                int inembed,
                int istride,
                int idist,
                int onembed,
                int ostride,
                int odist,
                int batch = 1);

    template<int Rank>
    FFTPlanBase(thrust::complex<T> * out,
                thrust::complex<T> * in,
                const int (&n)[Rank],
                const int (&inembed)[Rank],
                int istride,
                int idist,
                const int (&onembed)[Rank],
                int ostride,
                int odist,
                int batch = 1);

    explicit operator bool() const { return *_plan; }

    void execute() const;

protected:

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
                int rank,
                cufftType type);

    void * _out;
    void * _in;
    cufftType _type;
    std::shared_ptr<cufftHandle> _plan;
};

}}}}

#define ISCE_CUDA_FFT_DETAIL_FFTPLANBASE_ICC
#include "FFTPlanBase.icc"
#undef ISCE_CUDA_FFT_DETAIL_FFTPLANBASE_ICC
