#include "RangeComp.h"
#include <complex>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

using namespace isce3::focus;
namespace py = pybind11;

void addbinding(pybind11::enum_<RangeComp::Mode> & pyMode)
{
    pyMode
        .value("Full", RangeComp::Mode::Full, R"(
        The output contains the full discrete convolution of the input with
        the matched filter.

        For an input signal and chirp of size N and M, the output size is
        (N + M - 1).)")

        .value("Valid", RangeComp::Mode::Valid, R"(
        The output contains only the valid discrete convolution of the input
        with the matched filter.

        For an input signal and chirp of size N and M, the output size is
        (max(M, N) - min(M, N) + 1).
        )")

        .value("Same", RangeComp::Mode::Same, R"(
        The output contains the discrete convolution of the input with the
        matched filter, cropped to the same size as the input signal.
        )");
}

void addbinding(py::class_<RangeComp>& pyRangeComp)
{
    using T = std::complex<float>;
    using chirp_t = std::vector<T>;
    using buf_t = py::array_t<T, py::array::c_style>;

    pyRangeComp
        .def(py::init<const chirp_t &, int, int, RangeComp::Mode>(),
            py::arg("chirp"), py::arg("inputsize"),
            py::arg("maxbatch") = 1, py::arg("mode") = RangeComp::Mode::Full,
            R"(
    Forms a matched filter from the time-reversed complex conjugate of the
    chirp replica and creates FFT plans for frequency domain convolution
    with the matched filter.

    chirp     Time-domain replica of the transmitted chirp waveform
    inputsize Number of range samples in the signal to be compressed
    maxbatch  Max batch size
    mode      Convolution output mode
            )")

        .def("rangecompress",
            [](RangeComp & self, buf_t & out, const buf_t & in) {
                if (in.ndim() != out.ndim())
                    throw std::length_error(
                        "require same ndim on input and output");
                int batch = 1;
                // XXX C++ method doesn't do any size/shape checks.
                // Require 2D with matching slow dim if batched operation.
                if (in.ndim() == 2) {
                    batch = in.shape(0);
                    if (in.shape(0) != out.shape(0))
                        throw std::length_error(
                            "require equal batch size on input and output");
                    if (in.shape(1) != self.inputSize())
                        throw std::length_error("unexpected input length");
                    if (out.shape(1) != self.outputSize())
                        throw std::length_error("unexpected output length");
                } else if (in.ndim() == 1) {
                    if (in.shape(0) != self.inputSize())
                        throw std::length_error("unexpected input length");
                    if (out.shape(0) != self.outputSize())
                        throw std::length_error("unexpected output length");
                } else {
                    throw std::invalid_argument("require 1D or 2D data");
                }
                self.rangecompress(out.mutable_data(), in.data(), batch);
            }, py::arg("out"), py::arg("in"), R"(
    Perform pulse compression on a batch of input signals

    Computes the frequency domain convolution of the input with the reference
    function.  Batch size inferred from first dimension of 2D data (1 for 1D).
            )")

        .def_property_readonly("chirp_size", &RangeComp::chirpSize)
        .def_property_readonly("input_size", &RangeComp::inputSize)
        .def_property_readonly("fft_size", &RangeComp::fftSize)
        // one word for symmetry with ctor argument
        .def_property_readonly("maxbatch", &RangeComp::maxBatch)
        .def_property_readonly("mode", &RangeComp::mode)
        .def_property_readonly("output_size", &RangeComp::outputSize)
        .def_property_readonly("first_valid_sample", &RangeComp::firstValidSample)

        .def("apply_notch", &RangeComp::applyNotch,
            py::arg("frequency"), py::arg("bandwidth") = 0.0, R"(
     Apply a notch to the baseband frequency-domain representation of the
     chirp reference function.
     
     Parameters
     ----------
     frequency : float
        Center frequency of the notch normalized by the sample rate, so should
        be in interval [-0.5, 0.5)
     bandwidth : float, default=0.0
        Total width for the notch taper.  If zero, then only the nearest FFT bin
        will be set to zero.
            )")

        .def("get_chirp_spectrum", [](const RangeComp& self) {
            return buf_t(self.fftSize(), self.chirp_spectrum());
        })
        ;
}
