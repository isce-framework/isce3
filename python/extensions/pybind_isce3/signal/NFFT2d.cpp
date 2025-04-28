#include "NFFT2d.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace isce3::signal;
namespace py = pybind11;

NFFT2dParams parse_nfft2d_params(const py::dict& params)
{
    auto parse_ms = [](const py::dict& d) {
        NFFTParams out;
        for (auto item : d) {
            auto key = item.first.cast<std::string>();
            if (key == "m") {
                out.m = item.second.cast<int>();
            }
            else if (key == "s") {
                out.s = item.second.cast<double>();
            }
            else {
                throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                    "unexpected NFFT keyword: " + key);
            }
        }
        return out;
    };
    NFFT2dParams out;
    for (auto item : params) {
        auto key = item.first.cast<std::string>();
        if (key == "rows") {
            out.rows = parse_ms(item.second.cast<py::dict>());
        }
        else if (key == "cols") {
            out.cols = parse_ms(item.second.cast<py::dict>());
        }
        else {
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "unexpected NFFT2dParms keyword: " + key);
        }
    }
    return out;
}

template<typename T>
void addbinding(py::class_<NFFT2d<T>>& pyNFFT2d)
{
    using dims_t = typename NFFT2d<T>::dims_t;
    pyNFFT2d
        .def(py::init<dims_t, dims_t, dims_t>(),
            py::arg("m"), py::arg("sizes"), py::arg("fft_sizes"))
        .def("interp", &NFFT2d<T>::interp,
            py::arg("t"), py::arg("periodic") = true)
        .def_property_readonly("spectrum", [](const NFFT2d<T>& self) {
            const auto ptr = self.spectrum();
            const auto dims = self.fft_sizes();
            // property implies reference_internal return value policy
            return py::array_t<std::complex<T>>(dims, ptr);
        })
        ;
        // TODO more methods
}

// instantiate
template void addbinding(py::class_<NFFT2d<float>>&);
template void addbinding(py::class_<NFFT2d<double>>&);

void addbinding_make_image_nfft2d(pybind11::module& m)
{
    // TODO generalize to CF32 and CF64
    using T = float;
    using array_t = isce3::core::EArray2D<std::complex<T>>;
    m.def("make_image_nfft2d", [](Eigen::Ref<array_t> image,
                                  py::dict params,
                                  bool pad_input)
        {
            const auto params_ = parse_nfft2d_params(params);
            return makeImageNFFT2d<T>(image, params_, pad_input);
        },
        py::arg("image"),
        py::arg("params") = py::dict{},
        py::arg("pad_input") = false)
    ;
}