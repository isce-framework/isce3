//
// Author: Joshua Cohen
// Copyright 2017
//

#include "Interpolator.h"

/** 1-D sinc evaluation */
template <class U, class V>
U isce3::core::
sinc_eval(const isce3::core::Matrix<U> & arr, const isce3::core::Matrix<V> & intarr, int idec,
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
isce3::core::
sinc_eval(const isce3::core::Matrix<std::complex<double>> &,
          const isce3::core::Matrix<double> &, int, int, double, int);

template std::complex<double>
isce3::core::
sinc_eval(const isce3::core::Matrix<std::complex<double>> &,
          const isce3::core::Matrix<float> &, int, int, double, int);

template std::complex<float>
isce3::core::
sinc_eval(const isce3::core::Matrix<std::complex<float>> &,
          const isce3::core::Matrix<double> &, int, int, double, int);

template std::complex<float>
isce3::core::
sinc_eval(const isce3::core::Matrix<std::complex<float>> &,
          const isce3::core::Matrix<float> &, int, int, double, int);

template double
isce3::core::
sinc_eval(const isce3::core::Matrix<double> &,
          const isce3::core::Matrix<double> &, int, int, double, int);

template double
isce3::core::
sinc_eval(const isce3::core::Matrix<double> &,
          const isce3::core::Matrix<float> &, int, int, double, int);

template float
isce3::core::
sinc_eval(const isce3::core::Matrix<float> &,
          const isce3::core::Matrix<double> &, int, int, double, int);

template float
isce3::core::
sinc_eval(const isce3::core::Matrix<float> &,
          const isce3::core::Matrix<float> &, int, int, double, int);

// end of file
