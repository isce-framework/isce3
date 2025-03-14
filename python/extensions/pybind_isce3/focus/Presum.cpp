#include "Presum.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include <isce3/core/Kernels.h>
#include <isce3/except/Error.h>
#include <isce3/focus/Presum.h>
#include <isce3/math/complexOperations.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/product/RadarGridParameters.h>

namespace py = pybind11;
using namespace isce3::focus;
using isce3::core::Kernel;
using isce3::math::complex_operations::unitPhasor;
using std::int64_t;


// see python docstring below
void apply_presum_weights(
    py::array_t<std::complex<float>>& out,
    const py::array_t<double>& pulse_times,
    const py::array_t<double>& doppler,
    const py::array_t<double>& weights,
    const py::array_t<std::complex<float>>& pulses)
{
    const auto m = pulse_times.size();
    const auto n = doppler.size();
    if ((pulse_times.ndim() != 1) or (doppler.ndim() != 1)) {
        throw std::length_error("Expected 1D pulse_times and doppler arrays.");
    }
    if (out.size() != n) {
        throw std::length_error("Output vector length doesn't match Doppler");
    }
    if ((weights.ndim() != 2) or (weights.shape(0) != m) or (weights.shape(1) != n)) {
        throw std::length_error(
            "Weight dimensions don't match time and Doppler vectors");
    }
    if ((pulses.ndim() != 2) or (pulses.shape(0) != m) or (pulses.shape(1) != n)) {
        throw std::length_error(
            "Raw data dimensions don't match time and Doppler vectors");
    }

    // trust me ;-)
    const auto t_ = pulse_times.unchecked<1>();
    const auto fd_ = doppler.unchecked<1>();
    const auto pulses_ = pulses.unchecked<2>();
    const auto weights_ = weights.unchecked<2>();
    auto out_ = out.mutable_unchecked<1>();

    // parallelize on range since typically n >> m and to avoid temporaries
    #pragma omp parallel for 
    for (auto j = 0; j < n; ++j) {
        auto sum = std::complex<float>(0.0f);
        for (auto i = 0; i < m; ++i) {
            const auto deramp = unitPhasor<float>(-2 * M_PI * t_(i) * fd_(j));
            const float weight = static_cast<float>(weights_(i, j));
            sum += weight * deramp * pulses_(i, j);
        }
        out_(j) = sum;
    }
}


// see Python docstring below
auto get_unique_ids(const py::array_t<int64_t>& ids)
{
    // Make a copy we can modify.  Use std::vector for easy STL usage.
    // Use a for loop to avoid requiring stride == sizeof(int64_t).
    const auto n = ids.size();
    std::vector<int64_t> unique_ids(n);
    auto ids_ = ids.unchecked<1>();
    for (auto i = 0; i < n; ++i) { unique_ids[i] = ids_(i); }

    // remove consecutive duplicates
    // numpy skips this step, see
    // https://github.com/numpy/numpy/blob/db4f43983cb938f12c311e1f5b7165e270c393b4/numpy/lib/arraysetops.py#L336
    auto last = std::unique(unique_ids.begin(), unique_ids.end());
    unique_ids.erase(last, unique_ids.end());

    // sort the smaller list
    std::sort(unique_ids.begin(), unique_ids.end());

    // now remove all duplicates
    last = std::unique(unique_ids.begin(), unique_ids.end());
    unique_ids.erase(last, unique_ids.end());

    // copy back to numpy array
    // avoid compiler warning about narrowing cast: since nu <= n there's no
    // way it can overflow
    const auto nu = static_cast<py::ssize_t>(unique_ids.size());
    auto retval = py::array_t<int64_t>(py::array::ShapeContainer{nu}, unique_ids.data());
    // It's not very well documented whether the above actually makes a copy,
    // so convince ourselves with some assertions.
    assert(retval.data() != unique_ids.data());
    assert(retval.owndata());
    return retval;
}


// python docstring below
auto compute_ids_from_mask(const py::array_t<bool>& mask)
{
    if (mask.ndim() != 2) {
        throw std::length_error("Expected 2D mask");
    }
    const auto num_ranges = mask.shape(0);
    const auto num_pulses = mask.shape(1);
    if (num_pulses > 63) {
        throw std::length_error("Hash algorithm requires number of pulses "
            "less than number of bits in identifier (64)");
    }
    py::array_t<int64_t> ids(num_ranges);

    // trust me ;-)
    auto ids_ = ids.mutable_unchecked<1>();
    auto mask_ = mask.unchecked<2>();

    #pragma omp parallel for
    for (auto i = 0; i < num_ranges; ++i) {
        ids_(i) = 0;
        for (auto j = 0; j < num_pulses; ++j) {
            if (mask_(i, j)) {
                ids_(i) |= 1 << j;
            }
        }
    }
    return ids;
}


void addbindings_presum(pybind11::module& m)
{
    m.def("get_presum_weights",
        [](const Kernel<double>& acorr,
           const Eigen::Ref<const Eigen::VectorXd>& t,
           double tout) {
               long offset = 0;
               auto w = getPresumWeights(acorr, t, tout, &offset);
               return std::make_pair(offset, w);
           },
        R"(Compute weights for reconstructing data from non-uniform samples.

        Parameters
        ----------
        acorr : isce3.core.Kernel
            Autocorrelation function (argument same units as t).
        t : array_like
            Times of available input data (sorted).
        tout : float
            Desired output time.

        Returns
        -------
        offset : int
            Index to first sample to multiply.
        weights : ndarray
            Weights to multiply into given data.

        Notes
        -----
        Sample reconstructed from input data `x` with
            weights.dot(x[offset:offset + len(weights)])
        )",
        py::arg("acorr"), py::arg("t"), py::arg("tout")
    )
    .def("fill_weights", &fillWeights)
    .def("apply_presum_weights", &apply_presum_weights,
        R"(Apply Doppler deramp and calculate weighted sum of pulses, hopefully
        faster than Python.  Specifically, calculate

        deramp = np.exp(-2j * np.pi * pulse_times[:, None] * doppler[None, :])
        out[:] = (weights * deramp * pulses).sum(axis=0)

        Parameters
        ----------
        out : np.ndarray[np.complex64] 
            Output complex64 vector storing weighted sum of pulses.
            shape (num_ranges,)
        pulse_times : np.ndarray[np.float64]
            Input pulse times relative to output point in seconds.
            shape (num_pulses,)
        doppler : np.ndarray[np.float64]
            Doppler in Hz for each range sample.
            shape (num_ranges,)
        weights : np.ndarray[np.float64]
            Pulse weights for all range samples.
            shape (num_pulses, num_ranges)
        pulses : np.ndarray[np.complex64]
            Raw echo data.
            shape (num_pulses, num_ranges)
        )", py::arg("out"), py::arg("pulse_times"), py::arg("doppler"),
        py::arg("weights"), py::arg("pulses")
    )
    .def("get_unique_ids", &get_unique_ids,
        R"(Equivalent to numpy.unique but hopefully faster, especially when
        input contains long runs of repeated values.

        Parameters
        ----------
        ids : np.ndarray[np.int64]
            Vector of identifiers

        Returns
        -------
        np.ndarray[np.int64]
            Vector of unique identifiers
        )", py::arg("ids")
    )
    .def("compute_ids_from_mask", &compute_ids_from_mask,
        R"(Compute identifier for each pattern of gaps in a valid data mask.

        Parameters
        ----------
        mask : np.ndarray[bool]
            Matrix storing True for every valid pixel.
            shape = (num_ranges, num_pulses)
            Require num_pulses < 64 (typically num_pulses is around 4 for NISAR)

        Returns
        -------
        np.ndarray[np.int64]
            Hash of gap pattern for each range bin, computed by setting bit j on
            for every valid pulse j.
            shape = (num_ranges,)
        )", py::arg("mask"))
    ;
}
