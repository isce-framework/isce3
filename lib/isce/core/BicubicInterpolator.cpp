//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Liang Yu, Bryan Riel
// Copyright 2017-2018

#include "Interpolator.h"

template <typename U>
isce::core::BicubicInterpolator<U>::
BicubicInterpolator() : isce::core::Interpolator<U>(isce::core::BICUBIC_METHOD) {
    // Set the weights of the 2D bicubic kernel
    _weights = {
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       -3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0,-2.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 0.0, 0.0,-2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,-3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0,-2.0, 0.0, 0.0,-1.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,-2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
       -3.0, 3.0, 0.0, 0.0,-2.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-3.0, 3.0, 0.0, 0.0,-2.0,-1.0, 0.0, 0.0,
        9.0,-9.0, 9.0,-9.0, 6.0, 3.0,-3.0,-6.0, 6.0,-6.0,-3.0, 3.0, 4.0, 2.0, 1.0, 2.0,
       -6.0, 6.0,-6.0, 6.0,-4.0,-2.0, 2.0, 4.0,-3.0, 3.0, 3.0,-3.0,-2.0,-1.0,-1.0,-2.0,
        2.0,-2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0,-2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
       -6.0, 6.0,-6.0, 6.0,-3.0,-3.0, 3.0, 3.0,-4.0, 4.0, 2.0,-2.0,-2.0,-2.0,-1.0,-1.0,
        4.0,-4.0, 4.0,-4.0, 2.0, 2.0,-2.0,-2.0, 2.0,-2.0,-2.0, 2.0, 1.0, 1.0, 1.0, 1.0
    };
}

/** @param[in] x X-coordinate to interpolate
  * @param[in] y Y-coordinate to interpolate
  * @param[in] z 2D matrix to interpolate. */
template <class U>
U
isce::core::BicubicInterpolator<U>::
interpolate(double x, double y, const isce::core::Matrix<U> & z) {

    const int x1 = std::floor(x);
    const int x2 = std::ceil(x);
    const int y1 = std::floor(y);
    const int y2 = std::ceil(y);

    const U denom = static_cast<U>(2.0);
    const U scale = static_cast<U>(0.25);

    // Future work: See "Future work" note from Interpolator::bilinear.
    const std::valarray<U> zz = {z(y1,x1), z(y1,x2), z(y2,x2), z(y2,x1)};

    // First order derivatives
    const std::valarray<U> dzdx = {
        (z(y1,x1+1) - z(y1,x1-1)) / denom,
        (z(y1,x2+1) - z(y1,x2-1)) / denom,
        (z(y2,x2+1) - z(y2,x2-1)) / denom,
        (z(y2,x1+1) - z(y2,x1-1)) / denom
    };
    const std::valarray<U> dzdy = {
        (z(y1+1,x1) - z(y1-1,x1)) / denom,
        (z(y1+1,x2+1) - z(y1-1,x2)) / denom,
        (z(y2+1,x2+1) - z(y2-1,x2)) / denom,
        (z(y2+1,x1+1) - z(y2-1,x1)) / denom
    };

    // Cross derivatives
    const std::valarray<U> dzdxy = {
        scale*(z(y1+1,x1+1) - z(y1-1,x1+1) - z(y1+1,x1-1) + z(y1-1,x1-1)),
        scale*(z(y1+1,x2+1) - z(y1-1,x2+1) - z(y1+1,x2-1) + z(y1-1,x2-1)),
        scale*(z(y2+1,x2+1) - z(y2-1,x2+1) - z(y2+1,x2-1) + z(y2-1,x2-1)),
        scale*(z(y2+1,x1+1) - z(y2-1,x1+1) - z(y2+1,x1-1) + z(y2-1,x1-1))
    };
      
    // Compute polynomial coefficients 
    std::valarray<U> q(16);
    for (int i = 0; i < 4; ++i) {
        q[i] = zz[i];
        q[i+4] = dzdx[i];
        q[i+8] = dzdy[i];
        q[i+12] = dzdxy[i];
    }

    // Matrix multiply by stored weights
    Matrix<U> c(4, 4);
    for (int i = 0; i < 16; ++i) {
        U qq(0.0);
        for (int j = 0; j < 16; ++j) {
            const U cpx_wt = static_cast<U>(_weights[i*16+j]);
            qq += cpx_wt * q[j];
        }
        c(i) = qq;
    }

    // Compute and normalize desired results
    const U t = x - x1;
    const U u = y - y1;
    U ret = 0.0;
    for (int i = 3; i >= 0; i--)
        ret = t*ret + ((c(i,3)*u + c(i,2))*u + c(i,1))*u + c(i,0);
    return ret;
}

// Forward declaration of classes
template class isce::core::BicubicInterpolator<double>;
template class isce::core::BicubicInterpolator<float>;
template class isce::core::BicubicInterpolator<std::complex<double>>;
template class isce::core::BicubicInterpolator<std::complex<float>>;

// end of file
