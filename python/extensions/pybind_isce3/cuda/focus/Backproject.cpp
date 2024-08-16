#include "Backproject.h"
#include "pybind_isce3/focus/Backproject.h"  // parse parameter dicts

#include <isce3/container/RadarGeometry.h>
#include <isce3/core/Kernels.h>
#include <isce3/except/Error.h>
#include <isce3/cuda/focus/Backproject.h>
#include <isce3/focus/DryTroposphereModel.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace isce3::cuda::focus;
using namespace isce3::except;

using isce3::container::RadarGeometry;
using isce3::core::Kernel;
using isce3::error::ErrorCode;
using isce3::focus::parseDryTropoModel;
using isce3::geometry::DEMInterpolator;

void addbinding_cuda_backproject(py::module& m)
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
                int batch,
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

            if (batch < 1) {
                throw DomainError(ISCE_SRCINFO(), "batch size must be > 0");
            }

            ErrorCode err;
            {
                py::gil_scoped_release release;
                err = backproject(out_data, out_geometry, in_data, in_geometry,
                        dem, fc, ds, kernel, atm, r2gparams, g2rparams, batch,
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
            py::arg("batch") = 1024,
            py::arg("height") = py::none());

    m.def("backproject_first_stage", [](
                const py::array_t<std::complex<float>, py::array::c_style> in,
                const RadarGeometry& in_geometry,
                const py::array_t<double>& in_azimuth_time,
                double range_bandwidth,
                const DEMInterpolator& dem,
                double fc,
                double ds,
                const Kernel<float>& kernel,
                const std::string& dry_tropo_model,
                py::dict rdr2geo_params,
                double oversample_range,
                double oversample_azimuth) {

            if (in.ndim() != 2) {
                throw InvalidArgument(ISCE_SRCINFO(), "input signal data must be 2-D");
            }

            if (in.shape()[0] != in_geometry.gridLength() or
                in.shape()[1] != in_geometry.gridWidth()) {

                std::string errmsg = "input signal data shape must match "
                    "input radar grid shape";
                throw InvalidArgument(ISCE_SRCINFO(), errmsg);
            }

            DryTroposphereModel atm = parseDryTropoModel(dry_tropo_model);

            const auto r2gparams = parse_rdr2geo_params(rdr2geo_params);

            // TODO avoid copy
            std::vector<double> aztime(in_azimuth_time.data(),
                in_azimuth_time.data() + in_azimuth_time.size());

            const std::complex<float>* in_data = in.data();

            auto [err, grid, outp, heightp] = [&]() {
                py::gil_scoped_release release;
                return isce3::cuda::focus::backprojectFirstStage(in_data,
                    in_geometry, aztime, range_bandwidth,
                    dem, fc, ds, kernel, atm, r2gparams,
                    oversample_range, oversample_azimuth);
            }();

            // TODO bind ErrorCode class.  For now return nonzero on failure.
            bool status = err == ErrorCode::Success;
            // TODO verify that this ctor takes ownership of data pointer!
            auto bytes = sizeof(std::complex<float>);
            auto out = py::array_t<std::complex<float>>(
                {grid.length(), grid.width()}, {grid.width() * bytes, bytes},
                outp.release());
            bytes = sizeof(float);
            auto height = py::array_t<float>(
                {grid.length(), grid.width()}, {grid.width() * bytes, bytes},
                heightp.release());
            return std::make_tuple(status, grid, out, height);
            },
            R"(
                Focus in azimuth via time-domain backprojection.
            )",
            py::arg("in"),
            py::arg("in_geometry"),
            py::arg("in_azimuth_time"),
            py::arg("range_bandwidth"),
            py::arg("dem"),
            py::arg("fc"),
            py::arg("ds"),
            py::arg("kernel"),
            py::arg("dry_tropo_model") = "tsx",
            py::arg("rdr2geo_params") = py::dict(),
            py::arg("oversample_range") = 1.2,
            py::arg("oversample_azimuth") = 1.2);

    m.def("project_polar_to_geo", [](
                py::array_t<std::complex<float>>& geo_image,
                const py::array_t<double>& geo_points,
                const isce3::focus::PolarGrid& grid,
                const py::array_t<std::complex<float>>& polar_image,
                const double wavelength,
                py::dict nfft2_params) {

            // get root finding parameters
            const auto params = parse_nfft2_params(nfft2_params);
            if (geo_points.size() != 3 * geo_image.size()) {
                throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "shape mismatch between geo image and position arrays");
            }
            auto n = static_cast<size_t>(geo_image.size());
            if ((polar_image.shape(0) != grid.length())
                    or (polar_image.shape(1) != grid.width())) {
                throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "shape mismatch between polar image array and grid");
            }

            // XXX type cast after checking sizes, assume alignment is okay
            // TODO redo with Eigen::Map or change interface from Vec3 to double[3]?
            using isce3::core::Vec3;
            static_assert(sizeof(Vec3) == (sizeof(double[3])));
            const auto ptr = reinterpret_cast<const Vec3*>(geo_points.data());

            auto status = isce3::cuda::focus::projectPolarToGeo(geo_image.mutable_data(), ptr,
                n, grid, polar_image.data(), wavelength, params);

            if (status != ErrorCode::Success) {
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                    "Could not compute map projection of polar grid coords.");
            }
        },
        py::arg("geo_image"),
        py::arg("geo_points"),
        py::arg("grid"),
        py::arg("polar_image"),
        py::arg("wavelength"),
        py::arg("nfft2_params") = py::dict());
}
