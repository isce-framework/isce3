#include "Constants.h"
#include <isce/core/Constants.h>

namespace py = pybind11;

void add_constants(py::module & core)
{
    py::enum_<isce::core::dataInterpMethod>(core, "dataInterpMethod")
        .value("sinc", isce::core::SINC_METHOD)
        .value("bilinear", isce::core::BILINEAR_METHOD)
        .value("bicubic", isce::core::BICUBIC_METHOD)
        .value("nearest", isce::core::NEAREST_METHOD)
        .value("biquintic", isce::core::BIQUINTIC_METHOD);
        // nicer not to export_values() to parent namespace
}
