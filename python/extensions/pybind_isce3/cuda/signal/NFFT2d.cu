#include "NFFT2d.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind_isce3/signal/NFFT2d.h>

using namespace isce3::cuda::signal;
namespace py = pybind11;

template<typename T>
void addbinding(py::class_<NFFT2d<T>>& pyNFFT2d)
{
    using dims_t = typename NFFT2d<T>::dims_t;
    pyNFFT2d
        .def(py::init<dims_t, dims_t, dims_t>(),
            py::arg("m"), py::arg("sizes"), py::arg("fft_sizes"))
        ;
        // TODO more methods
}

template<typename T>
void addbinding(py::class_<NFFT2dResult<T>>& pyNFFT2dResult)
{
    using dims_t = typename NFFT2d<T>::dims_t;
    pyNFFT2dResult
        .def(py::init<const isce3::signal::NFFT2dResult<T>&>())
        .def("copy_to_host", [](const NFFT2dResult<T>& self) {
            return isce3::signal::NFFT2dResult<T>(self);
        })
        ;
}

// instantiate
template void addbinding(py::class_<NFFT2d<float>>&);
template void addbinding(py::class_<NFFT2d<double>>&);
template void addbinding(py::class_<NFFT2dResult<float>>&);
template void addbinding(py::class_<NFFT2dResult<double>>&);

void addbinding_make_image_nfft2d_gpu(pybind11::module& m)
{
    // TODO generalize to CF32 and CF64
    using T = float;
    using array_t = isce3::core::EArray2D<std::complex<T>>;
    m.def("make_image_nfft2d", [](Eigen::Ref<array_t> image,
                                  py::dict params,
                                  bool pad_input)
        {
            const auto params_ = parse_nfft2d_params(params);
            return isce3::cuda::signal::makeImageNFFT2d<T>(image, params_,
                pad_input);
        },
        py::arg("image"),
        py::arg("params") = py::dict{},
        py::arg("pad_input") = false)
    ;
}