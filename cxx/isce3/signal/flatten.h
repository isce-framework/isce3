#pragma once

#include <isce3/core/EMatrix.h>

namespace isce3 { namespace signal {

/**
 * Flattens an interferogram by removing the phase component due to slant
 * range difference (i.e, slant range offset) between the interferometric
 * pair.
 *
 * \param[in,out] ifgram      input/output interferogram ro be flattened
 * \param[in] range_offset    slant range offset [pixels]
 * \param[in] range_spacing   slant range spacing [meters]
 * \param[in] wavelength      radar wavelength [meters]
 */
void flatten(
        Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> ifgram,
        const Eigen::Ref<const isce3::core::EArray2D<double>>& range_offset,
        double range_spacing, double wavelength);

}} // namespace isce3::signal
