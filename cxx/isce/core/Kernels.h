#pragma once

#include "forward.h"

#include <cmath>
#include <valarray>

#include <isce/math/Bessel.h>

namespace isce { namespace core {

/** Abstract base class for all kernels.
 *
 * Basically just a closure around an arbitrary function and a width property.
 * Template parameter T defines type for coefficients, typically float or
 * double.
 */
template<typename T>
class Kernel {
public:
    Kernel(double width) : _halfwidth(fabs(width / 2.0)) {}

    /** Virtual destructor (allow destruction of base Kernel pointer) */
    virtual ~Kernel() {}

    /** Evaluate kernel at given location in [-halfwidth, halfwidth] */
    virtual T operator()(double x) const = 0;

    /** Get width of kernel.
     *
     * Units are the same as are used for calls to operator().
     */
    double width() const { return _halfwidth * 2; }

protected:
    double _halfwidth;
};

/** Bartlett kernel (triangle function). */
template<typename T>
class BartlettKernel : public Kernel<T> {
public:
    /** Triangle function constructor. */
    BartlettKernel(double width) : Kernel<T>(width) {}

    T operator()(double x) const override;
};

/** Linear kernel, which is just a special case of Bartlett. */
template<typename T>
class LinearKernel : public BartlettKernel<T> {
public:
    LinearKernel() : BartlettKernel<T>(2.0) {}
};

/** Kernel based on the paper by Knab for interpolating band-limited signals.
 *
 * For details see references
 * @cite knab1983
 * @cite migliaccio2007
 */
template<typename T>
class KnabKernel : public Kernel<T> {
public:
    /** Constructor of Knab's kernel.
     *
     * @param[in] width     Total width of kernel.
     * @param[in] bandwidth Bandwidth of signal to be interpolated, as a
     *                      fraction of the sample rate (0 < bandwidth < 1).
     */
    KnabKernel(double width, double bandwidth)
        : Kernel<T>(width), _bandwidth(bandwidth)
    {}

    T operator()(double x) const override;

    /** Get bandwidth of kernel. */
    double bandwidth() const { return _bandwidth; }

private:
    double _bandwidth;
};

/** NFFT time-domain kernel.
 *
 * This is called \f$ \phi(x) \f$ in the NFFT papers @cite keiner2009 ,
 * specifically the Kaiser-Bessel window function.
 * The domain is scaled so that usage is the same as other ISCE kernels, e.g.,
 * for x in [0,n) instead of [-0.5,0.5).
 */
template<typename T>
class NFFTKernel : public Kernel<T> {
public:
    /** Constructor of NFFT kernel.
     *
     * @param[in] m         Half kernel size (width = 2*m+1)
     * @param[in] n         Length of input signal.
     * @param[in] fft_size  FFT Transform size (> n).
     */
    NFFTKernel(int m, int n, int fft_size);

    T operator()(double x) const override;

private:
    int _m;
    int _n;
    int _fft_size;
    T _scale;
    T _b;
};

/** Tabulated Kernel */
template<typename T>
class TabulatedKernel : public Kernel<T> {
public:
    /** Constructor of tabulated kernel.
     *
     * @param[in] kernel    Kernel to sample.
     * @param[in] n         Table size.
     */
    template<typename Tin>
    TabulatedKernel(const Kernel<Tin>& kernel, int n);

    T operator()(double x) const override;

private:
    std::valarray<T> _table;
    int _imax;
    T _1_dx;
};

/** Polynomial Kernel */
template<typename T>
class ChebyKernel : public Kernel<T> {
public:
    /** Constructor that computes fit of another Kernel.
     *
     * @param[in] kernel    Kernel to fit (assumed even).
     * @param[in] n         Number of coefficients.
     */
    template<typename Tin>
    ChebyKernel(const Kernel<Tin>& kernel, int n);

    T operator()(double x) const override;

private:
    std::valarray<T> _coeffs;
    T _scale;
};

}} // namespace isce::core

#include "Kernels.icc"
