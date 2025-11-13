#include "Constants.h"
#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>

namespace py = pybind11;

void add_constants(py::module & core)
{
    py::enum_<isce3::core::dataInterpMethod>(core, "DataInterpMethod")
        .value("SINC", isce3::core::SINC_METHOD)
        .value("BILINEAR", isce3::core::BILINEAR_METHOD)
        .value("BICUBIC", isce3::core::BICUBIC_METHOD)
        .value("NEAREST", isce3::core::NEAREST_METHOD)
        .value("BIQUINTIC", isce3::core::BIQUINTIC_METHOD);
        // nicer not to export_values() to parent namespace

    core.attr("speed_of_light") = py::float_(isce3::core::speed_of_light);
    core.attr("earth_spin_rate") = py::float_(isce3::core::EarthSpinRate);
    core.attr("GLOBAL_MIN_HEIGHT") = py::float_(isce3::core::GLOBAL_MIN_HEIGHT);
    core.attr("GLOBAL_MAX_HEIGHT") = py::float_(isce3::core::GLOBAL_MAX_HEIGHT);
    core.attr("WGS84_ELLIPSOID") = isce3::core::Ellipsoid();
    core.attr("SINC_HALF") = isce3::core::SINC_HALF;

    // TODO agree on LUT postings.
    core.attr("RSLC_CALIBRATION_SECTION_SAMPLING") = 50;

    // Extra margin added to RSLC LUTs to ensure that, after
    // geocoding with an interpolation algoritm (e.g., bicubic spline),
    // the LUTs fully cover the geocoded imagery extents.
    core.attr("RSLC_LUTS_EXTRA_MARGIN_IN_PIXELS") = 11;
}
