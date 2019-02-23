//
// Author: Brian Hawkins
// Copyright 2019
//

#include "Interp1d.h"

template <typename TD, typename TK>
isce::core::Interp1d<TD,TK>
isce::core::Interp1d<TD,TK>::
Linear()
{
    std::shared_ptr<isce::core::BartlettKernel<TK>>
        p = std::make_shared<isce::core::BartlettKernel<TK>>(1.0);
    return isce::core::Interp1d<TD,TK>(p);
}

template <typename TD, typename TK>
isce::core::Interp1d<TD,TK>
isce::core::Interp1d<TD,TK>::
Knab(double width, double bandwidth)
{
    std::shared_ptr<isce::core::KnabKernel<TK>>
        p = std::make_shared<isce::core::KnabKernel<TK>>(width, bandwidth);
    return isce::core::Interp1d<TD,TK>(p);
}

template <typename TD, typename TK>
TD
isce::core::Interp1d<TD,TK>::
interp(std::valarray<TD> &x, double t)
{
    long i0 = 0;
    if (_width % 2 == 0) {
        i0 = (long) ceil(t);
    } else {
        i0 = (long) round(t);
    }
    long low = i0 - _width/2;  // integer division implicit floor()
    long high = low + _width;
    typename std::common_type<TD,TK>::type sum = 0;
    if ((low < 0) || (high >= x.size())) {
        // XXX log/throw error?
        return sum;
    }
    for (long i=low; i<high; i++) {
        double ti = i - t;
        TK w = (*_kernel)(ti);
        sum += w * x[i];
    }
    return sum;
}

template class isce::core::Interp1d<float,float>;
template class isce::core::Interp1d<std::complex<float>,float>;
template class isce::core::Interp1d<double,double>;
template class isce::core::Interp1d<std::complex<double>,double>;
