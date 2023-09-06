/*
 * @file cuSincOverSampler.h
 * @brief A class performs sinc interpolation/oversampling
 *
 * Oversample a given 2d signal by i_covs factor.
 * Only signals within(-i_sincwindow, i_sincwindow) are oversampled
 * The interpolation zone may also be shifted, if the max location is not at the center.
 *
 * The sinc interpolation is based on the formula
 *   $$x(t) = \sum_{n=-\infty}^{\infty} x_n f( \Omega_c t-n )$$
 *   with $f(x) = \text{sinc}(x)$, or a complex filter
 *   such as the sinc(x) convoluted with Hamming Window used here.
 *   In practice, a finite length of n (i_intplength) is used for interpolation.
 *
 * @note most parameters are currently hardwired; you need to change
 *    the source code below if you need to adjust the parameters.
 */

#pragma once

// dependencites
#include "cuArrays.h"

namespace isce3::matchtemplate::pycuampcor {

class cuSincOverSamplerR2R
{
 private:
    static const int i_sincwindow = 2;
    ///< the oversampling is only performed within \pm i_sincwindow*i_covs around the peak
    static const int i_weight = 1;       ///< weight for cos() pedestal
    const float r_pedestal = 0.0f;       ///< height of pedestal
    const float r_beta = 0.75f;          ///< a low-band pass
    const float r_relfiltlen = 6.0f;     ///< relative filter length

    static const int i_decfactor = 4096; ///< max decimals between original grid to set up the sinc kernel

    int i_covs;         ///< oversampling factor
    int i_intplength;   ///< actual filter length = r_relfiltlen/r_beta
    int i_filtercoef;   //< length of the sinc kernel i_intplength*i_decfactor+1

    float * r_filter;   // sinc kernel with size i_filtercoef

 public:
    // constructor
    cuSincOverSamplerR2R(const int i_covs_);
    // set up sinc interpolation coefficients
    void cuSetupSincKernel();
    // execute interface
    void execute(cuArrays<float> *imagesIn, cuArrays<float> *imagesOut, cuArrays<int2> *center, int oversamplingFactor);
    // destructor
    ~cuSincOverSamplerR2R();
};

} // namespace
