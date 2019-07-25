//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

#include <isce/core/Poly1d.h>
#include "Bessel.h"
#include <cmath>

namespace isce{
    namespace math{

//Sourced from:
//https://github.com/boostorg/math/blob/master/include/boost/math/special_functions/detail/bessel_i0.hpp
double bessel_i0(double x)
{
    if (x == 0)
    {
        return 1;
    }

    if (x < 0)
    {
        return bessel_i0(-x);
    }

    if (x < 7.75)
    {
        // Bessel I0 over[10 ^ -16, 7.75]
        // Max error in interpolated form : 3.042e-18
        // Max Error found at double precision = Poly : 5.106609e-16 Cheb : 5.239199e-16
        isce::core::Poly1d bessel_i0_P0(14, 0.0, 1.0);
        bessel_i0_P0.coeffs[0] = 1.00000000000000000e+00;
        bessel_i0_P0.coeffs[1] = 2.49999999999999909e-01;
        bessel_i0_P0.coeffs[2] = 2.77777777777782257e-02;
        bessel_i0_P0.coeffs[3] = 1.73611111111023792e-03;
        bessel_i0_P0.coeffs[4] = 6.94444444453352521e-05;
        bessel_i0_P0.coeffs[5] = 1.92901234513219920e-06;
        bessel_i0_P0.coeffs[6] = 3.93675991102510739e-08;
        bessel_i0_P0.coeffs[7] = 6.15118672704439289e-10;
        bessel_i0_P0.coeffs[8] = 7.59407002058973446e-12;
        bessel_i0_P0.coeffs[9] = 7.59389793369836367e-14;
        bessel_i0_P0.coeffs[10] = 6.27767773636292611e-16;
        bessel_i0_P0.coeffs[11] = 4.34709704153272287e-18;
        bessel_i0_P0.coeffs[12] = 2.63417742690109154e-20;
        bessel_i0_P0.coeffs[13] = 1.13943037744822825e-22;
        bessel_i0_P0.coeffs[14] = 9.07926920085624812e-25;
        
        double a = x * x / 4;
        return a * bessel_i0_P0.eval(a) + 1;
    }
    else if(x < 500)
    {
        isce::core::Poly1d bessel_i0_P1(21, 0.0, 1.0);
        bessel_i0_P1.coeffs[0] = 3.98942280401425088e-01;
        bessel_i0_P1.coeffs[1] = 4.98677850604961985e-02;
        bessel_i0_P1.coeffs[2] = 2.80506233928312623e-02;
        bessel_i0_P1.coeffs[3] = 2.92211225166047873e-02;
        bessel_i0_P1.coeffs[4] = 4.44207299493659561e-02;
        bessel_i0_P1.coeffs[5] = 1.30970574605856719e-01;
        bessel_i0_P1.coeffs[6] = -3.35052280231727022e+00;
        bessel_i0_P1.coeffs[7] = 2.33025711583514727e+02;
        bessel_i0_P1.coeffs[8] = -1.13366350697172355e+04;
        bessel_i0_P1.coeffs[9] = 4.24057674317867331e+05;
        bessel_i0_P1.coeffs[10] = -1.23157028595698731e+07;
        bessel_i0_P1.coeffs[11] = 2.80231938155267516e+08;
        bessel_i0_P1.coeffs[12] = -5.01883999713777929e+09;
        bessel_i0_P1.coeffs[13] = 7.08029243015109113e+10;
        bessel_i0_P1.coeffs[14] = -7.84261082124811106e+11;
        bessel_i0_P1.coeffs[15] = 6.76825737854096565e+12;
        bessel_i0_P1.coeffs[16] = -4.49034849696138065e+13;
        bessel_i0_P1.coeffs[17] = 2.24155239966958995e+14;
        bessel_i0_P1.coeffs[18] = -8.13426467865659318e+14;
        bessel_i0_P1.coeffs[19] = 2.02391097391687777e+15;
        bessel_i0_P1.coeffs[20] = -3.08675715295370878e+15;
        bessel_i0_P1.coeffs[21] = 2.17587543863819074e+15;
        
        return std::exp(x) * bessel_i0_P1.eval(1.0 / x) / std::sqrt(x);
    }
    else
    {
        // Max error in interpolated form : 2.437e-18
        // Max Error found at double precision = Poly : 1.216719e-16
        isce::core::Poly1d bessel_i0_P2(4, 0.0, 1.0);
        bessel_i0_P2.coeffs[0] = 3.98942280401432905e-01;
        bessel_i0_P2.coeffs[1] = 4.98677850491434560e-02;
        bessel_i0_P2.coeffs[2] = 2.80506308916506102e-02;
        bessel_i0_P2.coeffs[3] = 2.92179096853915176e-02;
        bessel_i0_P2.coeffs[4] = 4.53371208762579442e-02;

        double ex = std::exp(x / 2);
        double result = ex * bessel_i0_P2.eval(1.0 / x) / std::sqrt(x);
        result *= ex;
        return result;
    }
}

}
}

