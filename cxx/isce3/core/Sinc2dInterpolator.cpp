//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Liang Yu, Bryan Riel
// Copyright 2017-2018

#include <algorithm>
#include "Interpolator.h"

/** \param[in] kernelLength     Length of sinc kernel
  * \param[in] decimationFactor Sinc decimation factor */
template <typename U>
isce3::core::Sinc2dInterpolator<U>::
Sinc2dInterpolator(int kernelLength, int decimationFactor) :
    isce3::core::Interpolator<U>(SINC_METHOD),
    _decimationFactor{decimationFactor},
    _kernelLength{kernelLength},
    _halfKernelLength{kernelLength / 2} {

    // Temporary valarray for storing sinc coefficients
    std::valarray<double> filter(0.0, decimationFactor * kernelLength);
    _sinc_coef(1.0, kernelLength, decimationFactor, 0.0, 1, filter);

    // Resize member kernel matrix
    _kernel.resize(decimationFactor, kernelLength);

    // Normalize filter
    for (size_t i = 0; i < decimationFactor; ++i) {
        // Compute filter sum
        double ssum = 0.0;
        for (size_t j = 0; j < kernelLength; ++j) {
            ssum += filter[i + decimationFactor*j];
        }
        // Normalize the filter coefficients and copy to transposed member kernel
        for (size_t j = 0; j < kernelLength; ++j) {
            filter[i + decimationFactor*j] /= ssum;
            _kernel(i,j) = filter[i + decimationFactor*j];
        }
    }
}

/** @param[in] x X-coordinate to interpolate
  * @param[in] y Y-coordinate to interpolate
  * @param[in] z 2D matrix to interpolate. */
template<class U>
U isce3::core::Sinc2dInterpolator<U>::interp_impl(double x, double y,
                                                 const Map& z) const
{

    // Separate interpolation coordinates into integer and fractional components
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    const double fx = x - ix;
    const double fy = y - iy;

    // Check edge conditions
    U interpVal(0.0);
    if ((ix < (_halfKernelLength - 1)) || (ix > (z.cols() - _halfKernelLength - 1)))
        return interpVal;
    if ((iy < (_halfKernelLength - 1)) || (iy > (z.rows() - _halfKernelLength - 1)))
        return interpVal;

    // Modify integer interpolation coordinates for sinc evaluation
    const int xx = ix + _halfKernelLength;
    const int yy = iy + _halfKernelLength;

    // Call sinc interpolator
    interpVal = _sinc_eval_2d(z, xx, yy, fx, fy);
    return interpVal;
}

template<class U>
U isce3::core::Sinc2dInterpolator<U>::_sinc_eval_2d(const Map& arrin, int intpx,
                                                   int intpy, double frpx,
                                                   double frpy) const
{

    // Initialize return value
    U ret(0.0);

    // Get nearest kernel indices
    int ifracx = std::min(std::max(0, int(frpx*_decimationFactor)), _decimationFactor-1);
    int ifracy = std::min(std::max(0, int(frpy*_decimationFactor)), _decimationFactor-1);

    // Compute weighted sum from kernel
    for (int i = 0; i < _kernelLength; i++) {
        for (int j = 0; j < _kernelLength; j++) {
            ret += arrin(intpy-i,intpx-j) *
                   static_cast<U>(_kernel(ifracy,i)) *
                   static_cast<U>(_kernel(ifracx,j));
        }
    }

    // Done
    return ret;
}

template<class U>
void isce3::core::Sinc2dInterpolator<U>::_sinc_coef(
        double beta, double, int decfactor, double pedestal, int weight,
        std::valarray<double>& filter) const
{

    int filtercoef = int(filter.size());
    double wgthgt = (1.0 - pedestal) / 2.0;
    double soff = (filtercoef - 1.) / 2.;

    double wgt, s, fct;
    for (int i = 0; i < filtercoef; i++) {
        wgt = (1. - wgthgt) + (wgthgt * std::cos((M_PI * (i - soff)) / soff));
        s = (std::floor(i - soff) * beta) / (1. * decfactor);
        fct = ((s != 0.) ? (std::sin(M_PI * s) / (M_PI * s)) : 1.);
        filter[i] = ((weight == 1) ? (fct * wgt) : fct);
    }
}

// Forward declaration of classes
template class isce3::core::Sinc2dInterpolator<double>;
template class isce3::core::Sinc2dInterpolator<float>;
template class isce3::core::Sinc2dInterpolator<std::complex<double>>;
template class isce3::core::Sinc2dInterpolator<std::complex<float>>;

// end of file
