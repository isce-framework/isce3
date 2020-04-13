#include "Constants.h"
#include <isce/core/Constants.h>

namespace py = pybind11;

void add_constants(py::module & core)
{
    py::enum_<isce::core::dataInterpMethod>(core, "DataInterpMethod")
        .value("SINC", isce::core::SINC_METHOD)
        .value("BILINEAR", isce::core::BILINEAR_METHOD)
        .value("BICUBIC", isce::core::BICUBIC_METHOD)
        .value("NEAREST", isce::core::NEAREST_METHOD)
        .value("BIQUINTIC", isce::core::BIQUINTIC_METHOD);
        // nicer not to export_values() to parent namespace
}
