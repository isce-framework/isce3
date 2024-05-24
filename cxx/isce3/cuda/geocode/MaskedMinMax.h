#include <thrust/device_vector.h>
#include <thrust/pair.h>

namespace isce3::cuda::geocode {

/** Calculate min and max of data vector with mask applied to it.
 *
 * \returns     min_max_pair    Pair of double values where first is minimum
 *                              and second in maximum. If all values masked,
 *                              a pair of NaNs are returned.
 * \param[in]   data            Double vector to retrieve minimum and maximum
 *                              from. Must be same size as mask and cannot be
 *                              emtpy.
 * \param[in]   mask            Bool vector applied to data using Numpy masked
 *                              array convention where true is masked and false
 *                              is unmasked. Must be same size as data and
 *                              cannot be empty.
 *
 */
thrust::pair<double, double> masked_minmax(
        const thrust::device_vector<double>& data,
        const thrust::device_vector<bool>& mask);

} // namespace isce3::cuda::geocode
