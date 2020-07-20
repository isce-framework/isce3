#pragma once

#include "forward.h"

#include <cmath>
#include <thrust/device_vector.h>

#include <isce3/core/Common.h>
#include <isce3/core/Kernels.h>

namespace isce3 { namespace cuda { namespace core {

/**
 * CRTP base class for kernels
 *
 * \tparam T Kernel coefficients value type
 */
template<typename T, class Derived>
class Kernel {
public:
    /** Kernel coefficients value type */
    using value_type = T;

    /** Construct a new Kernel object. */
    explicit constexpr Kernel(double width) noexcept
        : _halfwidth(std::abs(width) / 2.)
    {}

    /**
     * Get the halfwidth of the kernel.
     *
     * Units are the same as are used for calls to \p operator().
     */
    constexpr double halfwidth() const noexcept { return _halfwidth; }

    /**
     * Get the width of the kernel.
     *
     * Units are the same as are used for calls to \p operator().
     */
    constexpr double width() const noexcept { return 2. * _halfwidth; }

    /** Evaluate the kernel at a given location in [-halfwidth, halfwidth]. */
    CUDA_HOSTDEV T operator()(double t) const;

private:
    double _halfwidth;
};

/** Bartlett kernel (triangle function) */
template<typename T>
class BartlettKernel : public Kernel<T, BartlettKernel<T>> {
    using Base = Kernel<T, BartlettKernel<T>>;
    friend Base;

public:
    /** A non-owning kernel view type that can be passed to device code */
    using view_type = BartlettKernel<T>;

    /** Construct a new BartlettKernel object. */
    explicit constexpr BartlettKernel(double width) : Base(width) {}

    /** Construct from corresponding host kernel object */
    BartlettKernel(const isce3::core::BartlettKernel<T>& other)
        : BartlettKernel(other.width())
    {}

protected:
    /** \internal Implementation of \p operator() */
    constexpr T eval(double t) const;
};

/** Linear kernel (special case of Bartlett) */
template<typename T>
class LinearKernel : public BartlettKernel<T> {
    using Base = BartlettKernel<T>;

public:
    /** A non-owning kernel view type that can be passed to device code */
    using view_type = LinearKernel<T>;

    /** Construct a new LinearKernel object. */
    constexpr LinearKernel() : Base(2.) {}

    /** Construct from corresponding host kernel object */
    LinearKernel(const isce3::core::LinearKernel<T>&) : LinearKernel() {}
};

/**
 * Kernel based on the paper by Knab for interpolating band-limited signals
 * @cite knab1983
 * @cite migliaccio2007
 */
template<typename T>
class KnabKernel : public Kernel<T, KnabKernel<T>> {
    using Base = Kernel<T, KnabKernel<T>>;
    friend Base;

public:
    /** A non-owning kernel view type that can be passed to device code */
    using view_type = KnabKernel<T>;

    /**
     * Construct a new KnabKernel object.
     *
     * \param[in] width     Kernel width
     * \param[in] bandwidth Bandwidth of the signal to be interpolated, as a
     *                      fraction of the sample rate (0 < bandwidth < 1).
     */
    constexpr KnabKernel(double width, double bandwidth);

    /** Construct from corresponding host kernel object */
    KnabKernel(const isce3::core::KnabKernel<T>& other)
        : Base(other.width()), _bandwidth(other.bandwidth())
    {}

    /** Get bandwidth of the kernel. */
    constexpr double bandwidth() const noexcept { return _bandwidth; }

protected:
    /** \internal Implementation of \p operator() */
    CUDA_HOSTDEV T eval(double t) const;

private:
    double _bandwidth;
};

/** A non-owning reference to a TabulatedKernel object */
template<typename T>
class TabulatedKernelView : public Kernel<T, TabulatedKernelView<T>> {
    using Base = Kernel<T, TabulatedKernelView<T>>;
    friend Base;

    friend class TabulatedKernel<T>;

public:
    /** A non-owning kernel view type that can be passed to device code */
    using view_type = TabulatedKernelView<T>;

    TabulatedKernelView(const TabulatedKernel<T>& kernel);

    /**
     * Evaluate the kernel at a given location in [-halfwidth, halfwidth].
     *
     * \internal Override the base method to make it __device__-only
     */
    CUDA_DEV T operator()(double t) const { return eval(t); }

protected:
    /** \internal Implementation of \p operator() */
    CUDA_DEV T eval(double t) const;

private:
    const T* _table;
    int _imax;
    T _rdx;
};

/** Tabulated kernel */
template<typename T>
class TabulatedKernel : public Kernel<T, TabulatedKernel<T>> {
    using Base = Kernel<T, TabulatedKernel<T>>;
    friend Base;

    friend class TabulatedKernelView<T>;

public:
    /** A non-owning kernel view type that can be passed to device code */
    using view_type = TabulatedKernelView<T>;

    /**
     * Construct a new TabulatedKernel object.
     *
     * The input kernel is assumed to be even.
     *
     * \param[in] kernel Kernel to sample
     * \param[in] n      Table size
     */
    template<class OtherKernel>
    TabulatedKernel(const OtherKernel& kernel, int n);

    /** Construct from corresponding host kernel object */
    TabulatedKernel(const isce3::core::TabulatedKernel<T>&);

    /**
     * Evaluate the kernel at a given location in [-halfwidth, halfwidth].
     *
     * \internal Override the base method to make it __device__-only
     */
    CUDA_DEV T operator()(double t) const { return eval(t); }

protected:
    /** \internal Implementation of \p operator() */
    CUDA_DEV T eval(double t) const;

private:
    thrust::device_vector<T> _table;
    int _imax;
    T _rdx;
};

/** A non-owning reference to a ChebyKernel object */
template<typename T>
class ChebyKernelView : public Kernel<T, ChebyKernelView<T>> {
    using Base = Kernel<T, ChebyKernelView<T>>;
    friend Base;

    friend class ChebyKernel<T>;

public:
    /** A non-owning kernel view type that can be passed to device code */
    using view_type = ChebyKernelView<T>;

    ChebyKernelView(const ChebyKernel<T>& kernel);

    /**
     * Evaluate the kernel at a given location in [-halfwidth, halfwidth].
     *
     * \internal Override the base method to make it __device__-only
     */
    CUDA_DEV T operator()(double t) const { return eval(t); }

protected:
    /** \internal Implementation of \p operator() */
    CUDA_DEV T eval(double t) const;

private:
    const T* _coeffs;
    int _n;
    T _scale;
};

/** Chebyshev polynomial kernel */
template<typename T>
class ChebyKernel : public Kernel<T, ChebyKernel<T>> {
    using Base = Kernel<T, ChebyKernel<T>>;
    friend Base;

    friend class ChebyKernelView<T>;

public:
    /** A non-owning kernel view type that can be passed to device code */
    using view_type = ChebyKernelView<T>;

    /**
     * Construct a new ChebyKernel object by computing a fit to another kernel.
     *
     * The input kernel is assumed to be even.
     *
     * \param[in] kernel Kernel to fit
     * \param[in] n      Number of coefficients
     */
    template<class OtherKernel>
    ChebyKernel(const OtherKernel& kernel, int n);

    /** Construct from corresponding host kernel object */
    ChebyKernel(const isce3::core::ChebyKernel<T>& other)
        : Base(other.width()), _scale(4. / other.width()),
          _coeffs(other.coeffs())
    {}

    /**
     * Evaluate the kernel at a given location in [-halfwidth, halfwidth].
     *
     * \internal Override the base method to make it __device__-only
     */
    CUDA_DEV T operator()(double t) const { return eval(t); }

protected:
    /** \internal Implementation of \p operator() */
    CUDA_DEV T eval(double t) const;

private:
    T _scale;
    thrust::device_vector<T> _coeffs;
};

}}} // namespace isce3::cuda::core

#include "Kernels.icc"
