#pragma once

#include <complex>

#include "detail/FFTPlanBase.h"

namespace isce { namespace fft {

/** RAII wrapper encapsulating FFTW plan for forward FFT execution */
template<typename T>
class FwdFFTPlan final : public detail::FFTPlanBase<FFTW_FORWARD, T> {
public:
    using super_t = detail::FFTPlanBase<FFTW_FORWARD, T>;
    using super_t::super_t;

    /**
     * Construct an invalid plan.
     *
     * The plan is allocated but not initialized. It should not be executed.
     */
    FwdFFTPlan() : super_t() {}

    // the following are declared here just for the purpose of doxygenating them
    // they are actually defined in the base class
#if 0
    /**
     * 1-D complex-to-complex forward transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    FwdFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               int n,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D complex-to-complex forward transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    FwdFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               const int (&n)[Rank],
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * 1-D complex-to-complex forward transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] nembed Size of the array(s) containing \p in & \p out
     * \param[in] stride Stride between adjacent elements in the input/output
     * \param[in] dist Stride between adjacent batches in the input/output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    FwdFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               int n,
               int nembed,
               int stride,
               int dist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D complex-to-complex forward transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] nembed Shape of the array(s) containing \p in & \p out
     * \param[in] stride Stride between adjacent elements in the input/output
     * \param[in] dist Stride between adjacent batches in the input/output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    FwdFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               const int (&n)[Rank],
               const int (&nembed)[Rank],
               int stride,
               int dist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * 1-D complex-to-complex forward transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] inembed Size of the array containing \p in
     * \param[in] istride Stride between adjacent elements in the input
     * \param[in] idist Stride between adjacent batches in the input
     * \param[in] onembed Size of the array containing \p out
     * \param[in] ostride Stride between adjacent elements in the output
     * \param[in] odist Stride between adjacent batches in the output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    FwdFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               int n,
               int inembed,
               int istride,
               int idist,
               int onembed,
               int ostride,
               int odist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D complex-to-complex forward transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] inembed Shape of the array containing \p in
     * \param[in] istride Stride between adjacent elements in the input
     * \param[in] idist Stride between adjacent batches in the input
     * \param[in] onembed Shape of the array containing \p out
     * \param[in] ostride Stride between adjacent elements in the output
     * \param[in] odist Stride between adjacent batches in the output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    FwdFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               const int (&n)[Rank],
               const int (&inembed)[Rank],
               int istride,
               int idist,
               const int (&onembed)[Rank],
               int ostride,
               int odist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);
#endif

    /**
     * 1-D real-to-complex forward transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    FwdFFTPlan(std::complex<T> * out,
               T * in,
               int n,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D real-to-complex forward transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    FwdFFTPlan(std::complex<T> * out,
               T * in,
               const int (&n)[Rank],
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * 1-D real-to-complex forward transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] nembed Size of the array(s) containing \p in & \p out
     * \param[in] stride Stride between adjacent elements in the input/output
     * \param[in] dist Stride between adjacent batches in the input/output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    FwdFFTPlan(std::complex<T> * out,
               T * in,
               int n,
               int nembed,
               int stride,
               int dist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D real-to-complex forward transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] nembed Shape of the array(s) containing \p in & \p out
     * \param[in] stride Stride between adjacent elements in the input/output
     * \param[in] dist Stride between adjacent batches in the input/output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    FwdFFTPlan(std::complex<T> * out,
               T * in,
               const int (&n)[Rank],
               const int (&nembed)[Rank],
               int stride,
               int dist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * 1-D real-to-complex forward transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] inembed Size of the array containing \p in
     * \param[in] istride Stride between adjacent elements in the input
     * \param[in] idist Stride between adjacent batches in the input
     * \param[in] onembed Size of the array containing \p out
     * \param[in] ostride Stride between adjacent elements in the output
     * \param[in] odist Stride between adjacent batches in the output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    FwdFFTPlan(std::complex<T> * out,
               T * in,
               int n,
               int inembed,
               int istride,
               int idist,
               int onembed,
               int ostride,
               int odist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D real-to-complex forward transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] inembed Shape of the array containing \p in
     * \param[in] istride Stride between adjacent elements in the input
     * \param[in] idist Stride between adjacent batches in the input
     * \param[in] onembed Shape of the array containing \p out
     * \param[in] ostride Stride between adjacent elements in the output
     * \param[in] odist Stride between adjacent batches in the output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    FwdFFTPlan(std::complex<T> * out,
               T * in,
               const int (&n)[Rank],
               const int (&inembed)[Rank],
               int istride,
               int idist,
               const int (&onembed)[Rank],
               int ostride,
               int odist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    // the following are declared here just for the purpose of doxygenating them
    // they are actually defined in the base class
#if 0
    /** True if the underlying plan is valid */
    explicit operator bool() const;

    /** Compute the transform. */
    void execute() const;
#endif
};

/** RAII wrapper encapsulating FFTW plan for inverse FFT execution */
template<typename T>
class InvFFTPlan final : public detail::FFTPlanBase<FFTW_BACKWARD, T> {
public:
    using super_t = detail::FFTPlanBase<FFTW_BACKWARD, T>;
    using super_t::super_t;

    /**
     * Construct an invalid plan.
     *
     * The plan is allocated but not initialized. It should not be executed.
     */
    InvFFTPlan() : super_t() {}

    // the following are declared here just for the purpose of doxygenating them
    // they are actually defined in the base class
#if 0
    /**
     * 1-D complex-to-complex inverse transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    InvFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               int n,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D complex-to-complex inverse transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    InvFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               const int (&n)[Rank],
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * 1-D complex-to-complex inverse transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] nembed Size of the array(s) containing \p in & \p out
     * \param[in] stride Stride between adjacent elements in the input/output
     * \param[in] dist Stride between adjacent batches in the input/output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    InvFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               int n,
               int nembed,
               int stride,
               int dist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D complex-to-complex inverse transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] nembed Shape of the array(s) containing \p in & \p out
     * \param[in] stride Stride between adjacent elements in the input/output
     * \param[in] dist Stride between adjacent batches in the input/output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    InvFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               const int (&n)[Rank],
               const int (&nembed)[Rank],
               int stride,
               int dist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * 1-D complex-to-complex inverse transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] inembed Size of the array containing \p in
     * \param[in] istride Stride between adjacent elements in the input
     * \param[in] idist Stride between adjacent batches in the input
     * \param[in] onembed Size of the array containing \p out
     * \param[in] ostride Stride between adjacent elements in the output
     * \param[in] odist Stride between adjacent batches in the output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    InvFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               int n,
               int inembed,
               int istride,
               int idist,
               int onembed,
               int ostride,
               int odist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D complex-to-complex inverse transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] inembed Shape of the array containing \p in
     * \param[in] istride Stride between adjacent elements in the input
     * \param[in] idist Stride between adjacent batches in the input
     * \param[in] onembed Shape of the array containing \p out
     * \param[in] ostride Stride between adjacent elements in the output
     * \param[in] odist Stride between adjacent batches in the output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    InvFFTPlan(std::complex<T> * out,
               std::complex<T> * in,
               const int (&n)[Rank],
               const int (&inembed)[Rank],
               int istride,
               int idist,
               const int (&onembed)[Rank],
               int ostride,
               int odist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);
#endif

    /**
     * 1-D complex-to-real inverse transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    InvFFTPlan(T * out,
               std::complex<T> * in,
               int n,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D complex-to-real inverse transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    InvFFTPlan(T * out,
               std::complex<T> * in,
               const int (&n)[Rank],
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * 1-D complex-to-real inverse transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] nembed Size of the array(s) containing \p in & \p out
     * \param[in] stride Stride between adjacent elements in the input/output
     * \param[in] dist Stride between adjacent batches in the input/output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    InvFFTPlan(T * out,
               std::complex<T> * in,
               int n,
               int nembed,
               int stride,
               int dist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D complex-to-real inverse transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] nembed Shape of the array(s) containing \p in & \p out
     * \param[in] stride Stride between adjacent elements in the input/output
     * \param[in] dist Stride between adjacent batches in the input/output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    InvFFTPlan(T * out,
               std::complex<T> * in,
               const int (&n)[Rank],
               const int (&nembed)[Rank],
               int stride,
               int dist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * 1-D complex-to-real inverse transform
     *
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Transform size
     * \param[in] inembed Size of the array containing \p in
     * \param[in] istride Stride between adjacent elements in the input
     * \param[in] idist Stride between adjacent batches in the input
     * \param[in] onembed Size of the array containing \p out
     * \param[in] ostride Stride between adjacent elements in the output
     * \param[in] odist Stride between adjacent batches in the output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    InvFFTPlan(T * out,
               std::complex<T> * in,
               int n,
               int inembed,
               int istride,
               int idist,
               int onembed,
               int ostride,
               int odist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    /**
     * N-D complex-to-real inverse transform
     *
     * \tparam Rank Transform dimensionality
     * \param[out] out Output buffer
     * \param[in,out] in Input data
     * \param[in] n Size of each transform dimension
     * \param[in] inembed Shape of the array containing \p in
     * \param[in] istride Stride between adjacent elements in the input
     * \param[in] idist Stride between adjacent batches in the input
     * \param[in] onembed Shape of the array containing \p out
     * \param[in] ostride Stride between adjacent elements in the output
     * \param[in] odist Stride between adjacent batches in the output
     * \param[in] batch Batch size
     * \param[in] flags FFTW plan creation flags
     * \param[in] threads Thread count for parallel execution
     */
    template<int Rank>
    InvFFTPlan(T * out,
               std::complex<T> * in,
               const int (&n)[Rank],
               const int (&inembed)[Rank],
               int istride,
               int idist,
               const int (&onembed)[Rank],
               int ostride,
               int odist,
               int batch = 1,
               unsigned flags = FFTW_MEASURE,
               int threads = 1);

    // the following are declared here just for the purpose of doxygenating them
    // they are actually defined in the base class
#if 0
    /** True if the underlying plan is valid */
    explicit operator bool() const;

    /** Compute the transform. */
    void execute() const;
#endif
};

}}

#define ISCE_FFT_FFTPLAN_ICC
#include "FFTPlan.icc"
#undef ISCE_FFT_FFTPLAN_ICC
