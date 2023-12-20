#include "Backproject.h"

#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <isce3/container/RadarGeometry.h>
#include <isce3/core/Kernels.h>
#include <isce3/except/Error.h>
#include <isce3/focus/Backproject.h>
#include <isce3/focus/DryTroposphereModel.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/detail/Rdr2Geo.h>

namespace py = pybind11;

using namespace isce3::focus;

using isce3::container::RadarGeometry;
using isce3::core::Kernel;
using isce3::error::ErrorCode;
using isce3::except::InvalidArgument;
using isce3::geometry::DEMInterpolator;
using isce3::geometry::detail::Rdr2GeoBracketParams;
using isce3::geometry::detail::Geo2RdrBracketParams;


Rdr2GeoBracketParams parse_rdr2geo_params(const py::dict& params)
{
    Rdr2GeoBracketParams out;
    for (auto item : params) {
        auto key = item.first.cast<std::string>();
        if (key == "tol_height") {
            out.tol_height = item.second.cast<double>();
        }
        else if (key == "look_min") {
            out.look_min = item.second.cast<double>();
        }
        else if (key == "look_max") {
            out.look_max = item.second.cast<double>();
        }
        else {
            throw InvalidArgument(ISCE_SRCINFO(),
                "unexpected rdr2geo_bracket keyword: " + key);
        }
    }
    return out;
}


Geo2RdrBracketParams parse_geo2rdr_params(const py::dict& params)
{
    Geo2RdrBracketParams out;
    for (auto item : params) {
        auto key = item.first.cast<std::string>();
        if (key == "tol_aztime") {
            out.tol_aztime = item.second.cast<double>();
        }
        else if (key == "time_start") {
            // don't combine with above to avoid throw on time_start=None
            if (not item.second.is_none()) {
                out.time_start = item.second.cast<double>();
            }
        }
        else if (key == "time_end") {
            // don't combine with above to avoid throw on time_end=None
            if (not item.second.is_none()) {
                out.time_end = item.second.cast<double>();
            }
        }
        else {
            throw InvalidArgument(ISCE_SRCINFO(),
                "unexpected geo2rdr_bracket keyword: " + key);
        }
    }
    return out;
}


void addbinding_backproject(py::module& m)
{
    m.def("backproject", [](
                py::array_t<std::complex<float>, py::array::c_style> out,
                const RadarGeometry& out_geometry,
                py::array_t<std::complex<float>, py::array::c_style> in,
                const RadarGeometry& in_geometry,
                const DEMInterpolator& dem,
                double fc,
                double ds,
                const Kernel<float>& kernel,
                const std::string& dry_tropo_model,
                py::dict rdr2geo_params,
                py::dict geo2rdr_params,
                std::optional<py::array_t<float, py::array::c_style>> height) {

            if (out.ndim() != 2) {
                throw InvalidArgument(ISCE_SRCINFO(), "output array must be 2-D");
            }

            if (out.shape()[0] != out_geometry.gridLength() or
                out.shape()[1] != out_geometry.gridWidth()) {

                std::string errmsg = "output array shape must match output "
                    "radar grid shape";
                throw InvalidArgument(ISCE_SRCINFO(), errmsg);
            }

            if (in.ndim() != 2) {
                throw InvalidArgument(ISCE_SRCINFO(), "input signal data must be 2-D");
            }

            if (in.shape()[0] != in_geometry.gridLength() or
                in.shape()[1] != in_geometry.gridWidth()) {

                std::string errmsg = "input signal data shape must match "
                    "input radar grid shape";
                throw InvalidArgument(ISCE_SRCINFO(), errmsg);
            }

            std::complex<float>* out_data = out.mutable_data();
            const std::complex<float>* in_data = in.data();
            float* height_data = nullptr;

            if (height.has_value()) {
                auto h = height.value();
                if (h.shape()[0] != out_geometry.gridLength() or
                    h.shape()[1] != out_geometry.gridWidth()) {

                    std::string errmsg = "height array shape must match output "
                        "radar grid shape";
                    throw InvalidArgument(ISCE_SRCINFO(), errmsg);
                }
                height_data = h.mutable_data();
            }

            DryTroposphereModel atm = parseDryTropoModel(dry_tropo_model);

            const auto r2gparams = parse_rdr2geo_params(rdr2geo_params);
            const auto g2rparams = parse_geo2rdr_params(geo2rdr_params);

            ErrorCode err;
            {
                py::gil_scoped_release release;
                err = backproject(out_data, out_geometry, in_data, in_geometry,
                    dem, fc, ds, kernel, atm, r2gparams, g2rparams,
                    height_data);
            }
            // TODO bind ErrorCode class.  For now return nonzero on failure.
            return err != ErrorCode::Success;
            },
            R"(
                Focus in azimuth via time-domain backprojection.
            )",
            py::arg("out"),
            py::arg("out_geometry"),
            py::arg("in"),
            py::arg("in_geometry"),
            py::arg("dem"),
            py::arg("fc"),
            py::arg("ds"),
            py::arg("kernel"),
            py::arg("dry_tropo_model") = "tsx",
            py::arg("rdr2geo_params") = py::dict(),
            py::arg("geo2rdr_params") = py::dict(),
            py::arg("height") = py::none());
}
