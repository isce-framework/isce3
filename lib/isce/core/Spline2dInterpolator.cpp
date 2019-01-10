//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Liang Yu, Bryan Riel
// Copyright 2017-2018

#include <pyre/journal.h>
#include "Interpolator.h"

/** @param[in] order Order of 2D spline */
template <typename U>
isce::core::Spline2dInterpolator<U>::
Spline2dInterpolator(size_t order) : 
    isce::core::Interpolator<U>(isce::core::BIQUINTIC_METHOD), 
    _order{order} {

    // Check validity of order
    if ((order < 3) || (order > 20)) {
        pyre::journal::error_t errorChannel("isce.core.Spline2dInterpolator");
        errorChannel
            << pyre::journal::at(__HERE__)
            << "Spline order must be between 3 and 20 "
            << "(received " + std::to_string(order) + ")"
            << pyre::journal::newline
            << pyre::journal::endl;
        return;
    }
}

/** @param[in] x X-coordinate to interpolate
  * @param[in] y Y-coordinate to interpolate
  * @param[in] z 2D matrix to interpolate. */
template <class U>
U
isce::core::Spline2dInterpolator<U>::
interpolate(double x, double y, const isce::core::Matrix<U> & z) {

    // Get array size
    const int nx = z.width();
    const int ny = z.length();

    // Get coordinates of start of spline window
    int i0, j0;
    if ((_order % 2) != 0) {
        i0 = y - 0.5;
        j0 = x - 0.5;
    } else {
        i0 = y;
        j0 = x;
    }
    i0 = i0 - (_order / 2) + 1;
    j0 = j0 - (_order / 2) + 1;

    std::valarray<U> A(_order), R(_order), Q(_order), HC(_order);
    
    for (int i = 0; i < _order; ++i) {
        const int indi = std::min(std::max(i0 + i, 0), ny - 2);
        for (int j = 0; j < _order; ++j) {
            const int indj = std::min(std::max(j0 + j, 0), nx - 2);
            A[j] = z(indi+1,indj+1);
        }
        _initSpline(A, _order, R, Q);
        HC[i] = _spline(x - j0, A, _order, R);
    }

    _initSpline(HC, _order, R, Q);
    return static_cast<U>(_spline(y - i0, HC, _order, R));
}

template <typename U>
U isce::core::Spline2dInterpolator<U>::
_spline(double x, const std::valarray<U> & Y, int n, const std::valarray<U> & R) {

    const U denom = static_cast<U>(6.0);
    if (x < 1.0) {
        return Y[0] + static_cast<U>(x - 1.0) * (Y[1] - Y[0] - (R[1] / denom));
    } else if (x > n) {
        return Y[n-1] + (static_cast<U>(x - n) * (Y[n-1] - Y[n-2] + (R[n-2] / denom)));
    } else {
        int j = int(std::floor(x));
        U xx = static_cast<U>(x - j);
        auto t0 = Y[j] - Y[j-1] - (R[j-1] / static_cast<U>(3.0)) - (R[j] / denom);
        auto t1 = xx * ((R[j-1] / static_cast<U>(2.0)) + (xx * ((R[j] - R[j-1]) / denom)));
        return Y[j-1] + (xx * (t0 + t1));
    }
}

template <typename U>
void isce::core::Spline2dInterpolator<U>::
_initSpline(const std::valarray<U> & Y, int n, std::valarray<U> & R,
            std::valarray<U> & Q) {
    Q[0] = U(0.0);
    R[0] = U(0.0);
    for (int i = 1; i < n - 1; ++i) {
        const U p = static_cast<U>(1.0) / 
                   (static_cast<U>(0.5) * Q[i-1] + static_cast<U>(2.0));
        Q[i] = static_cast<U>(-0.5) * p;
        R[i] = (static_cast<U>(3.0) * 
                (Y[i+1] - static_cast<U>(2.0) * Y[i] + Y[i-1]) - 
                 static_cast<U>(0.5) * R[i-1]) * p;
    }
    R[n-1] = U(0.0);
    for (int i = (n - 2); i > 0; --i)
        R[i] = Q[i] * R[i+1] + R[i];
}

// Forward declaration of classes
template class isce::core::Spline2dInterpolator<double>;
template class isce::core::Spline2dInterpolator<float>;
template class isce::core::Spline2dInterpolator<std::complex<double>>;
template class isce::core::Spline2dInterpolator<std::complex<float>>;

// end of file 
