//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018

#include "Interpolator.h"

/** @param[in] x X-coordinate to interpolate
  * @param[in] y Y-coordinate to interpolate
  * @param[in] z 2D matrix to interpolate. */
template <class U>
U
isce::core::NearestNeighborInterpolator<U>::
interpolate(double x, double y, const isce::core::Matrix<U> & z) {

    // Nearest indices
    const size_t row = static_cast<size_t>(std::round(y));
    const size_t col = static_cast<size_t>(std::round(x));

    // No bounds check yet
    return z(row, col);
}

// Forward declaration of classes
template class isce::core::NearestNeighborInterpolator<double>;
template class isce::core::NearestNeighborInterpolator<float>;
template class isce::core::NearestNeighborInterpolator<std::complex<double>>;
template class isce::core::NearestNeighborInterpolator<std::complex<float>>;

// end of file
