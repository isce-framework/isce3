//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Liang Yu, Bryan Riel
// Copyright 2017-2018

#include "Interpolator.h"

/** @param[in] x X-coordinate to interpolate
  * @param[in] y Y-coordinate to interpolate
  * @param[in] z 2D matrix to interpolate. */
template <class U>
U
isce::core::BilinearInterpolator<U>::
interpolate(double x, double y, const isce::core::Matrix<U> & z) {

    int x1 = std::floor(x);
    int x2 = std::ceil(x);
    int y1 = std::floor(y);
    int y2 = std::ceil(y);
    U q11 = z(y1,x1);
    U q12 = z(y2,x1);
    U q21 = z(y1,x2);
    U q22 = z(y2,x2);

    // Future work:
    // static_cast<> was applied below bc the compiler complained about things (complex
    // dtype probably), but the complex operators are overloaded to work with non-complex values on
    // lhs and rhs, so not sure why this wasn't working. In the future need to pull these out
    // (mostly just because of kludginess).
    if ((y1 == y2) && (x1 == x2)) {
        return q11;
    } else if (y1 == y2) {
        return (static_cast<U>((x2 - x) / (x2 - x1)) * q11) +
               (static_cast<U>((x - x1) / (x2 - x1)) * q21);
    } else if (x1 == x2) {
        return (static_cast<U>((y2 - y) / (y2 - y1)) * q11) +
               (static_cast<U>((y - y1) / (y2 - y1)) * q12);
    } else {
        return  ((q11 * static_cast<U>((x2 - x) * (y2 - y))) /
                 static_cast<U>((x2 - x1) * (y2 - y1))) +
                ((q21 * static_cast<U>((x - x1) * (y2 - y))) /
                 static_cast<U>((x2 - x1) * (y2 - y1))) +
                ((q12 * static_cast<U>((x2 - x) * (y - y1))) /
                 static_cast<U>((x2 - x1) * (y2 - y1))) +
                ((q22 * static_cast<U>((x - x1) * (y - y1))) /
                 static_cast<U>((x2 - x1) * (y2 - y1)));
    }
}

// Forward declaration of classes
template class isce::core::BilinearInterpolator<double>;
template class isce::core::BilinearInterpolator<float>;
template class isce::core::BilinearInterpolator<std::complex<double>>;
template class isce::core::BilinearInterpolator<std::complex<float>>;

// end of file
