#pragma once

#include <string>

namespace isce3::unwrap {

/**
 * Perform 2-D phase unwrapping using SNAPHU.
 *
 * \param[in] configfile Path to configuration file
 */
void snaphuUnwrap(const std::string& configfile);

} // namespace isce3::uwnrap
