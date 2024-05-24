#include <limits>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <isce3/except/Error.h>

#include "MaskedMinMax.h"

namespace isce3::cuda::geocode {

typedef thrust::tuple<double, bool> double_mask;

/** Comparison function used to find the minimum of double_mask list.
 * The double in the tuples are the values to evaluated with < operator. The
 * mask in the tuples is a bool that when true masks the double value from
 * < operator evaluation.
 *
 * \param[out]  bool    Whether or not left double is less than right double
 * with mask values taken into account. If both sides not masked, return left
 * double < right double. If only right masked, return true. Other cases return
 * false. \param[in]   lhs     Left double_mask tuple \param[in]   rhs     Right
 * double_mask tuple
 */
struct masked_min_compare {
    __host__ __device__ bool operator()(
            const double_mask lhs, const double_mask rhs)
    {
        // extract data values from tuples
        auto l_value = thrust::get<0>(lhs);
        auto r_value = thrust::get<0>(rhs);

        // extract mask values from tuples
        // true = mask data value
        // false = do not mask data value
        auto l_mask = thrust::get<1>(lhs);
        auto r_mask = thrust::get<1>(rhs);

        if (!l_mask && !r_mask)
            return l_value < r_value;
        else if (!l_mask && r_mask)
            return true;
        else
            return false;
    }
};

/** Comparison function used to find the maximum of double_mask list.
 * The double in the tuples are the values to evaluated with < operator. The
 * mask in the tuples is a bool that when true masks the double value from
 * < operator evaluation.
 *
 * \param[out]  bool    Whether or not left double is less than right double
 *                      with mask values taken into account. If both sides not
 *                      masked, return left double < right double. If only right
 *                      masked, return false. Other cases return true.
 * \param[in]   lhs     Left double_mask tuple
 * \param[in]   rhs     Right double_mask tuple
 */
struct masked_max_compare {
    __host__ __device__ bool operator()(
            const double_mask lhs, const double_mask rhs)
    {
        // extract data values from tuples
        auto l_value = thrust::get<0>(lhs);
        auto r_value = thrust::get<0>(rhs);

        // extract mask values from tuples
        // true = mask data value
        // false = do not mask data value
        auto l_mask = thrust::get<1>(lhs);
        auto r_mask = thrust::get<1>(rhs);

        if (!l_mask && !r_mask)
            return l_value < r_value;
        else if (!l_mask && r_mask)
            return false;
        else
            return true;
    }
};

thrust::pair<double, double> masked_minmax(
        const thrust::device_vector<double>& data,
        const thrust::device_vector<bool>& mask)
{
    if (data.empty()) {
        throw isce3::except::LengthError(ISCE_SRCINFO(), "Data vector empty");
    }

    if (mask.empty()) {
        throw isce3::except::LengthError(ISCE_SRCINFO(), "Mask vector empty");
    }

    if (data.size() != mask.size()) {
        throw isce3::except::LengthError(
                ISCE_SRCINFO(), "Data and mask vectors have different sizes");
    }

    // check if all values are masked
    size_t n_elem_masked = thrust::count(mask.begin(), mask.end(), true);

    //  if all values masked, return min and max as NaN
    if (n_elem_masked == mask.size()) {
        return thrust::pair<double, double>(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
    }

    auto masked_begin = thrust::make_zip_iterator(
            thrust::make_tuple(data.cbegin(), mask.cbegin()));
    auto masked_end = thrust::make_zip_iterator(
            thrust::make_tuple(data.cend(), mask.cend()));

    auto data_min = thrust::min_element(
            thrust::device, masked_begin, masked_end, masked_min_compare());
    auto data_max = thrust::max_element(
            thrust::device, masked_begin, masked_end, masked_max_compare());

    return thrust::pair<double, double>(
            thrust::get<0>(data_min[0]), thrust::get<0>(data_max[0]));
}

} // namespace isce3::cuda::geocode
