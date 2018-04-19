//
// Author: Joshua Cohen
// Copyright 2017
//

#include <iostream>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include "Constants.h"
#include "Interpolator.h"


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template <class U>
U isce::core::Interpolator::
bilinear(double x, double y, const Matrix<U> & z) {

    int x1 = std::floor(x);
    int x2 = std::ceil(x);
    int y1 = std::floor(y);
    int y2 = std::ceil(y);
    U q11 = z(y1,x1);
    U q12 = z(y2,x1);
    U q21 = z(y1,x2);
    U q22 = z(y2,x2);

    // Future work: static_cast<> was applied below bc the compiler complained about things (complex
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

template std::complex<double>
isce::core::Interpolator::
bilinear(double, double, const Matrix<std::complex<double>> &);

template std::complex<float>
isce::core::Interpolator::
bilinear(double, double, const Matrix<std::complex<float>> &);

template double
isce::core::Interpolator::
bilinear(double, double, const Matrix<double> &);

template float
isce::core::Interpolator::
bilinear(double, double, const Matrix<float> &);

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template <class U>
U isce::core::Interpolator::
bicubic(double x, double y, const Matrix<U> & z) {

    // The bicubic interpolation weights
    const std::valarray<double> weights = {
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
            const U cpx_wt = static_cast<U>(weights[i*16+j]);
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

template std::complex<double>
isce::core::Interpolator::
bicubic(double, double, const Matrix<std::complex<double>> &);

template std::complex<float>
isce::core::Interpolator::
bicubic(double, double, const Matrix<std::complex<float>> &);

template double
isce::core::Interpolator::
bicubic(double, double, const Matrix<double> &);

template float
isce::core::Interpolator::
bicubic(double, double, const Matrix<float> &);

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void isce::core::Interpolator::
sinc_coef(double beta, double relfiltlen, int decfactor, double pedestal, int weight,
          std::valarray<double> & filter) {

    int filtercoef = int(filter.size()) - 1;
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

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template <class U, class V>
U isce::core::Interpolator::
sinc_eval(const Matrix<U> & arr, const Matrix<V> & intarr, int idec,
          int intp, double frp, int nsamp) {
    U ret = 0.;
    int ilen = intarr.width();
    if ((intp >= (ilen-1)) && (intp < nsamp)) {
        int ifrc = std::min(std::max(0, int(frp*idec)), idec-1);
        for (int i=0; i<ilen; i++)
            ret += arr(intp-i) * static_cast<U>(intarr(ifrc,i));
    }
    return ret;
}

template std::complex<double>
isce::core::Interpolator::
sinc_eval(const Matrix<std::complex<double>> &,
          const Matrix<double> &, int, int, double, int);

template std::complex<double>
isce::core::Interpolator::
sinc_eval(const Matrix<std::complex<double>> &,
          const Matrix<float> &, int, int, double, int);

template std::complex<float>
isce::core::Interpolator::
sinc_eval(const Matrix<std::complex<float>> &,
          const Matrix<double> &, int, int, double, int);

template std::complex<float>
isce::core::Interpolator::
sinc_eval(const Matrix<std::complex<float>> &,
          const Matrix<float> &, int, int, double, int);

template double
isce::core::Interpolator::
sinc_eval(const Matrix<double> &,
          const Matrix<double> &, int, int, double, int);

template double
isce::core::Interpolator::
sinc_eval(const Matrix<double> &,
          const Matrix<float> &, int, int, double, int);

template float
isce::core::Interpolator::
sinc_eval(const Matrix<float> &,
          const Matrix<double> &, int, int, double, int);

template float
isce::core::Interpolator::
sinc_eval(const Matrix<float> &,
          const Matrix<float> &, int, int, double, int);


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template<class U, class V>
U isce::core::Interpolator::
sinc_eval_2d(const Matrix<U> & arrin, const Matrix<V> & intarr,
             int intpx, int intpy, double frpx, double frpy,
             int xlen, int ylen) {
    // Initialize return value
    U ret(0.0);
    int idec = intarr.length();
    int ilen = intarr.width();
    // Interpolate for valid indices
    if ((intpx >= (ilen-1)) && (intpx < xlen) && (intpy >= (ilen-1)) && (intpy < ylen)) {
        // Get nearest kernel indices
        int ifracx = std::min(std::max(0, int(frpx*idec)), idec-1);
        int ifracy = std::min(std::max(0, int(frpy*idec)), idec-1);
        // Compute weighted sum
        for (int i = 0; i < ilen; i++) {
            for (int j = 0; j < ilen; j++) {
                ret += arrin(intpx-i,intpy-j)
                     * intarr(ifracx,i)
                     * intarr(ifracy,j);
            }
        }
    }
    // Done
    return ret;
}

template std::complex<double>
isce::core::Interpolator::
sinc_eval_2d(const Matrix<std::complex<double>> &, const Matrix<double> &,
             int, int, double, double, int, int);

template std::complex<float>
isce::core::Interpolator::
sinc_eval_2d(const Matrix<std::complex<float>> &, const Matrix<float> &,
             int, int, double, double, int, int);

template double
isce::core::Interpolator::
sinc_eval_2d(const Matrix<double> &, const Matrix<double> &,
             int, int, double, double, int, int);

template double
isce::core::Interpolator::
sinc_eval_2d(const Matrix<double> &, const Matrix<float> &,
             int, int, double, double, int, int);

template float
isce::core::Interpolator::
sinc_eval_2d(const Matrix<float> &, const Matrix<double> &,
             int, int, double, double, int, int);

template float
isce::core::Interpolator::
sinc_eval_2d(const Matrix<float> &, const Matrix<float> &,
             int, int, double, double, int, int);

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template <class U>
U isce::core::Interpolator::
interp_2d_spline(int order, const Matrix<U> & z, double x, double y) {

    // Error checking
    if ((order < 3) || (order > 20)) {
        std::string errstr = "isce::core::Interpolator::interp_2d_spline - ";
        errstr += "Spline order must be between 3 and 20 ";
        errstr += "(received " + std::to_string(order) + ")";
        throw std::invalid_argument(errstr);
    }

    // Get array size
    const int nx = z.width();
    const int ny = z.length();

    // Get coordinates of start of spline window
    int i0, j0;
    if ((order % 2) != 0) {
        i0 = y - 0.5;
        j0 = x - 0.5;
    } else {
        i0 = y;
        j0 = x;
    }
    i0 = i0 - (order / 2) + 1;
    j0 = j0 - (order / 2) + 1;

    std::valarray<double> A(order), R(order), Q(order), HC(order);
    
    for (int i = 0; i < order; ++i) {
        const int indi = std::min(std::max(i0 + i, 0), ny - 2);
        for (int j = 0; j < order; ++j) {
            const int indj = std::min(std::max(j0 + j, 0), nx - 2);
            A[j] = z(indi+1,indj+1);
        }
        initSpline(A, order, R, Q);
        HC[i] = spline(x - j0, A, order, R);
    }

    initSpline(HC, order, R, Q);
    return static_cast<U>(spline(y - i0, HC, order, R));
}

template float
isce::core::Interpolator::
interp_2d_spline(int order, const Matrix<float> &z, double x, double y);

template double
isce::core::Interpolator::
interp_2d_spline(int order, const Matrix<double> &z, double x, double y);

void isce::core::
initSpline(const std::valarray<double> & Y, int n, std::valarray<double> & R,
           std::valarray<double> & Q) {
    Q[0] = 0.0;
    R[0] = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        const double p = 1.0 / (0.5 * Q[i-1] + 2.0);
        Q[i] = -0.5 * p;
        R[i] = (3 * (Y[i+1] - 2*Y[i] + Y[i-1]) - 0.5*R[i-1]) * p;
    }
    R[n-1] = 0.0;
    for (int i = (n - 2); i > 0; --i)
        R[i] = Q[i] * R[i+1] + R[i];
}

double isce::core::
spline(double x, const std::valarray<double> & Y, int n, const std::valarray<double> & R) {

    if (x < 1.0) {
        return Y[0] + (x - 1.0) * (Y[1] - Y[0] - (R[1] / 6.0));
    } else if (x > n) {
        return Y[n-1] + ((x - n) * (Y[n-1] - Y[n-2] + (R[n-2] / 6.)));
    } else {
        int j = int(std::floor(x));
        auto xx = x - j;
        auto t0 = Y[j] - Y[j-1] - (R[j-1] / 3.0) - (R[j] / 6.0);
        auto t1 = xx * ((R[j-1] / 2.0) + (xx * ((R[j] - R[j-1]) / 6)));
        return Y[j-1] + (xx * (t0 + t1));
    }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

double isce::core::Interpolator::
quadInterpolate(const std::valarray<double> & x, const std::valarray<double> & y, double xintp) {

    auto xin = xintp - x[0];
    std::valarray<double> x1(3), y1(3);
    for (int i=0; i<3; i++) {
        x1[i] = x[i] - x[0];
        y1[i] = y[i] - y[0];
    }
    double a = ((-y1[1] * x1[2]) + (y1[2] * x1[1])) 
        / ((-x1[2] * x1[1] * x1[1]) + (x1[1] * x1[2] * x1[2]));
    double b = (y1[1] - (a * x1[1] * x1[1])) / x1[1];
    return y[0] + (a * xin * xin) + (b * xin);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

double isce::core::Interpolator::
akima(int nx, int ny, const Matrix<float> & z, double x, double y) {

    Matrix<double> e(2, 2), sx(2, 2), sy(2, 2), sxy(2, 2);
    std::valarray<double> m(4);
    double wx2, wx3, wy2, wy3;
    int ix = x;
    int iy = y;

    wx2 = wx3 = wy2 = wy3 = 0.;
    for (int ii = 0; ii < 2; ii++) {
        int xx = std::min(std::max((ix+ii+1),3), (nx-2)) - 1;
        for (int jj = 0; jj < 2; jj++) {
            int yy = std::min(std::max((iy+jj+1),3), (ny-2)) - 1;

            m = {z(xx-1,yy) - z(xx-2,yy),
                 z(xx,yy) - z(xx-1,yy),
                 z(xx+1,yy) - z(xx,yy),
                 z(xx+2,yy) - z(xx+1,yy)};

            if ((std::abs(m[0] - m[1]) <= DBL_EPSILON) && (std::abs(m[2] - m[3]) <= DBL_EPSILON)) {
                sx(ii,jj) = 0.5 * (m[1] + m[2]);
            } else {
                wx2 = std::abs(m[3] - m[2]);
                wx3 = std::abs(m[1] - m[0]);
                sx(ii,jj) = ((wx2 * m[1]) + (wx3 * m[2])) / (wx2 + wx3);
            }

            m = {z(xx,yy-1) - z(xx,yy-2),
                 z(xx,yy) - z(xx,yy-1),
                 z(xx,yy+1) - z(xx,yy),
                 z(xx,yy+2) - z(xx,yy+1)};

            if ((std::abs(m[0] - m[1]) <= DBL_EPSILON) && (std::abs(m[2] - m[3]) <= DBL_EPSILON))
                sy(ii,jj) = 0.5 * (m[1] + m[2]);
            else {
                wy2 = std::abs(m[3] - m[2]);
                wy3 = std::abs(m[1] - m[0]);
                sy(ii,jj) = ((wy2 * m[1]) + (wy3 * m[2])) / (wy2 + wy3);
            }

            e(0,0) = m[1] - z(xx-1,yy) - z(xx-1,yy-1);
            e(0,1) = m[2] - z(xx-1,yy+1) - z(xx-1,yy);
            e(1,0) = z(xx+1,yy) - z(xx+1,yy-1) - m[1];
            e(1,1) = z(xx+1,yy+1) - z(xx+1,yy) - m[2];

            if ((std::abs(wx2) <= DBL_EPSILON) && (std::abs(wx3) <= DBL_EPSILON)) wx2 = wx3 = 1.;
            if ((std::abs(wy2) <= DBL_EPSILON) && (std::abs(wy3) <= DBL_EPSILON)) wy2 = wy3 = 1.;
            sxy(ii,jj) = ((wx2 * ((wy2 * e(0,0)) + (wy3 * e(0,1)))) +
                           (wx3 * ((wy2 * e(1,0)) + (wy3 * e(1,1))))) /
                           ((wx2 + wx3) * (wy2 + wy3));
        }
    }

    std::valarray<double> d = {
        (z(ix-1,iy-1) - z(ix,iy-1)) + (z(x,iy) - z(ix-1,iy)),
        (sx(0,0) + sx(1,0)) - (sx(1,1) + sx(0,1)),
        (sy(0,0) - sy(1,0)) - (sy(1,1) - sy(0,1)),
        (sxy(0,0) + sxy(1,0)) + (sxy(1,1) + sxy(0,1)),
        ((2 * sx(0,0)) + sx(1,0)) - (sx(1,1) + (2 * sx(0,1))),
        (2 * (sy(0,0) - sy(1,0))) - (sy(1,1) - sy(0,1)),
        (2 * (sxy(0,0) + sxy(1,0))) + (sxy(1,1) + sxy(0,1)),
        ((2 * sxy(0,0)) + sxy(1,0)) + (sxy(1,1) + (2 * sxy(0,1))),
        (2 * ((2 * sxy(0,0)) + sxy(1,0))) + (sxy(1,1) + (2 * sxy(0,1)))
    };

    std::valarray<double> poly = {
        (2 * ((2 * d[0]) + d[1])) + ((2 * d[2]) + d[3]),
        -((3 * ((2 * d[0]) + d[1])) + ((2 * d[5]) + d[6])),
        (2 * (sy(0,0) - sy(1,0))) + (sxy(0,0) + sxy(1,0)),
        (2 * (z(ix-1,iy-1) - z(ix,iy-1))) + (sx(0,0) + sx(1,0)),
        -((2 * ((3 * d[0]) + d[4])) + ((3 * d[2]) + d[7])),
        (3 * ((3 * d[0]) + d[4])) + ((3 * d[5]) + d[8]),
        -((3 * (sy(0,0) - sy(1,0))) + ((2 * sxy(0,0)) + sxy(1,0))),
        -((3 * (z(ix-1,iy-1) - z(ix,iy-1))) + ((2 * sx(0,0)) + sx(1,0))),
        (2 * (sx(0,0) - sx(0,1))) + (sxy(0,0) + sxy(0,1)),
        -((3 * (sx(0,0) - sx(0,1))) + ((2 * sxy(0,0)) + sxy(0,1))),
        sxy(0,0),
        sx(0,0),
        (2 * (z(ix-1,iy-1) - z(ix-1,iy))) + (sy(0,0) + sy(0,1)),
        -((3 * (z(ix-1,iy-1) - z(ix-1,iy))) + ((2 * sy(0,0)) + sy(0,1))),
        sy(0,0),
        z(ix-1,iy-1)
    };

    m[0] = (((((poly[0] * (y - iy)) + poly[1]) * (y - iy)) + poly[2]) * (y - iy)) + poly[3];
    m[1] = (((((poly[4] * (y - iy)) + poly[5]) * (y - iy)) + poly[6]) * (y - iy)) + poly[7];
    m[2] = (((((poly[8] * (y - iy)) + poly[9]) * (y - iy)) + poly[10]) * (y - iy)) + poly[11];
    m[3] = (((((poly[12] * (y - iy)) + poly[13]) * (y - iy)) + poly[14]) * (y - iy)) + poly[15];
    return (((((m[0] * (x - ix)) + m[1]) * (x - ix)) + m[2]) * (x - ix)) + m[3];
}
