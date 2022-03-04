#include "BinarySearchFunc.h"

#include <algorithm>

namespace isce3 { namespace antenna { namespace detail {

Eigen::Index bisect_right(const Eigen::Ref<const Eigen::ArrayXd>& x, double x0)
{
    auto x_begin = x.data();
    auto x_end = x.data() + x.size();
    auto it_up = std::upper_bound(x_begin, x_end, x0);
    return (it_up != x_end) ? std::distance(x_begin, it_up) : x.size() - 1;
}

Eigen::Index bisect_left(const Eigen::Ref<const Eigen::ArrayXd>& x, double x0)
{
    auto x_begin = x.data();
    auto x_end = x.data() + x.size();
    auto it_low = std::lower_bound(x_begin, x_end, x0);
    return (it_low != x_end) ? std::distance(x_begin, it_low) : x.size() - 1;
}

Eigen::Index locate_nearest(
        const Eigen::Ref<const Eigen::ArrayXd>& x, double x0)
{
    auto idx_right = bisect_left(x, x0);
    auto idx_left = (idx_right > 0) ? idx_right - 1 : 0;
    auto dif_right = std::fabs(x(idx_right) - x0);
    auto dif_left = std::fabs(x(idx_left) - x0);
    return (dif_left < dif_right) ? idx_left : idx_right;
}

ArrayXui locate_nearest(const Eigen::Ref<const Eigen::ArrayXd>& x,
        const Eigen::Ref<const Eigen::ArrayXd>& x0)
{
    ArrayXui idx_vec(x0.size());
    for (Eigen::Index idx = 0; idx < x0.size(); ++idx)
        idx_vec(idx) = locate_nearest(x, x0(idx));
    return idx_vec;
}

tuple4i_t intersect_nearest(const Eigen::Ref<const Eigen::ArrayXd>& x1,
        const Eigen::Ref<const Eigen::ArrayXd>& x2)
{
    Eigen::Index idx1_first {0};
    Eigen::Index idx2_first {0};
    Eigen::Index idx1_last {x1.size() - 1};
    Eigen::Index idx2_last {x2.size() - 1};

    if (x2(0) > x1(0))
        idx1_first = locate_nearest(x1, x2(0));
    else
        idx2_first = locate_nearest(x2, x1(0));

    if (x2(idx2_last) < x1(idx1_last))
        idx1_last = locate_nearest(x1, x2(idx2_last));
    else
        idx2_last = locate_nearest(x2, x1(idx1_last));

    return {idx1_first, idx1_last, idx2_first, idx2_last};
}

}}} // namespace isce3::antenna::detail
