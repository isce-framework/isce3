#include "Chirp.h"

#include <complex>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

#include <isce/focus/Chirp.h>

using namespace isce::focus;
namespace py = pybind11;

void addbinding_chirp(py::module& m)
{
    m.def("form_linear_chirp", &formLinearChirp, py::arg("chirprate"),
          py::arg("duration"), py::arg("samplerate"),
          py::arg("centerfreq") = 0.0, py::arg("amplitude") = 1.0,
          py::arg("phi") = 0.0, "Construct a time-domain LFM chirp waveform");
}
