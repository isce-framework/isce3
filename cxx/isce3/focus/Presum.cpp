
#include "Presum.h"

namespace isce3 { namespace focus {

Eigen::MatrixXd fillWeights(
        const Eigen::Ref<const Eigen::Array<long, Eigen::Dynamic, 1>>& ids,
        const std::unordered_map<long, const Eigen::Ref<const Eigen::VectorXd>>&
                lut)
{
    // length of id vector
    const auto nid = ids.size();
    // Extract first element to determine length of weight vectors.
    auto it = lut.begin();
    if (it == lut.end()) {
        throw std::length_error("Need at least one weight vector in LUT.");
    }
    const auto w0 = it->second;
    const auto nw = w0.size();
    // Allocate matrix and fill columns using LUT.
    Eigen::MatrixXd out(nw, nid);
    _Pragma("omp parallel for")
    for (long i = 0; i < nid; ++i) {
        const auto wi = lut.at(ids[i]);
        if (wi.size() != nw) {
            throw std::length_error("Weight vector lengths must all match.");
        }
        out.col(i) = wi;
    }
    return out;
}

}} // namespace isce3::focus
