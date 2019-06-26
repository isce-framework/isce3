//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-

#include "Interpolator.h"

/*
 * Returns the cubic-interpolated value between the middle two
 * of four evenly spaced points.
 *
 * This is equivalent to a uniform Catmull-Rom spline with evenly spaced points.
 * https://en.wikipedia.org/wiki/Centripetal_Catmull–Rom_spline
 *
 * The interpolation parameter is generalized to any spacing of points via
 * the parameter tfrac, which goes from 0 at p1 to 1 at p2.
 *
 * Derived using the following Mathematica snippet:

       (t2 - t)/(t2 - t1) B1 + (t - t1)/(t2 - t1) B2 //.
{B1 -> (t2 - t)/(t2 - t0) A1 + (t - t0)/(t2 - t0) A2,
 B2 -> (t3 - t)/(t3 - t1) A2 + (t - t1)/(t3 - t1) A3,
 A1 -> (t1 - t)/(t1 - t0) P0 + (t - t0)/(t1 - t0) P1,
 A2 -> (t2 - t)/(t2 - t1) P1 + (t - t1)/(t2 - t1) P2,
 A3 -> (t3 - t)/(t3 - t2) P2 + (t - t2)/(t3 - t2) P3,
 t1 -> t0 + dt,
 t2 -> t1 + dt,
 t3 -> t2 + dt,
 t -> tfrac*dt + t1};
CForm @ FullSimplify @ %

 */
template<typename T>
T cubicInterpolate(T p0, T p1, T p2, T p3, const double tfrac) {
    const auto tconj = 1. - tfrac;
    return (T(tfrac)*(p2 - p0*T(tconj*tconj) + (p2*T(tconj*3. + 1.) - p3*T(tconj))*T(tfrac)) +
            p1*T(tfrac*tfrac*(tfrac*3. - 5.) + 2.))/T(2.);
}

template <typename U>
isce::core::BicubicInterpolator<U>::
BicubicInterpolator() : isce::core::Interpolator<U>(isce::core::BICUBIC_METHOD) {}

/** @param[in] x X-coordinate to interpolate
  * @param[in] y Y-coordinate to interpolate
  * @param[in] z 2D matrix to interpolate. */
template <class U>
U
isce::core::BicubicInterpolator<U>::
interpolate(double x, double y, const isce::core::Matrix<U> & z) {
    // the closest pixel to the point of interest
    const int x0 = std::floor(x);
    const int y0 = std::floor(y);

    // Compute intermediate interpolation values
    U intp[4];
    for (int i = -1; i < 3; i++) {
        intp[i+1] = cubicInterpolate<U>(z(y0+i, x0-1),
                                        z(y0+i, x0  ),
                                        z(y0+i, x0+1),
                                        z(y0+i, x0+2), x-x0);
    }
    // Compute final result
    return cubicInterpolate<U>(intp[0], intp[1], intp[2], intp[3], y - y0);
}

// Forward declaration of classes
template class isce::core::BicubicInterpolator<double>;
template class isce::core::BicubicInterpolator<float>;
template class isce::core::BicubicInterpolator<std::complex<double>>;
template class isce::core::BicubicInterpolator<std::complex<float>>;

// end of file
