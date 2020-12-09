#include "PresumWeights.h"

#include <isce3/core/Kernels.h>
#include <isce3/except/Error.h>
#include <isce3/focus/PresumWeights.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace isce3::focus;
using isce3::core::Kernel;

void addbinding_get_presum_weights(pybind11::module& m)
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
    );
}
