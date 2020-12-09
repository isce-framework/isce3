#pragma once
/** @file PresumWeights.h */

#include <isce3/core/forward.h>
#include <Eigen/Dense>

namespace isce3 { namespace focus {

/** Get weight vector needed to reconstruct a sample at the given time.
 *
 * Given a set of data sampled at times xin, we want to compute a data point at
 * an arbitrary time xout.  This routine returns a set of coefficients such
 * that the output can be computed as
 *
 * \f$ y(x_{out}) = \sum_{i=0}^{N-1} w[i] \; y[i+{\rm offset}] \f$
 *
 * where `N` is the length of the weight vector. This method is known as Best
 * Linear Unbiased (BLU) reconstruction in the context of SweepSAR (aka
 * Staggered SAR) data @cite villano2013, or Wiener filtering in more general
 * contexts.
 *
 * The `offset` and `N` will be guaranteeed not to cause indexing errors in the
 * sum above, truncating the weight vector if needed. This means one should
 * always supply all available sample times, or otherwise be very careful when
 * breaking a data set into blocks.
 *
 * Also note that N may be zero when the gap is too large (e.g., there are no
 * correlated samples).
 *
 * @tparam KernelType One of the kernels available in isce3/core/Kernels.h
 *
 * @param[in]  acorr  Autocorrelation function (same time units as xin).
 *                    For SAR azimuth data, it's the Fourier transform of the
 *                    two-way azimuth antenna pattern (squared).  For the common
 *                    sinc antenna pattern model, the ACF is a piecewise cubic.
 * @param[in]  xin    Available sample times, monotonically increasing.
 * @param[in]  xout   Desired output sample time.
 * @param[out] offset Index to first non-zero weight.
 * @returns Weight vector.
 */
template<typename KernelType>
auto
getPresumWeights(const KernelType& acorr,
                 const Eigen::Ref<const Eigen::VectorXd>& xin, double xout,
                 long* offset);


template<typename KernelType>
auto
getPresumWeights(const KernelType& acorr,
                 const std::vector<double>& xin, double xout,
                 long* offset);

}}

#include "PresumWeights.icc"
