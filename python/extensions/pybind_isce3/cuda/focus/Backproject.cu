#include "Backproject.h"
#include "pybind_isce3/container/TypedPythonSequence.h"
#include "pybind_isce3/focus/Backproject.h"  // parse parameter dicts
#include "pybind_isce3/signal/NFFT2d.h"  // parse NFFT2d parameters

#include <isce3/container/RadarGeometry.h>
#include <isce3/core/Kernels.h>
#include <isce3/core/Vector.h>
#include <isce3/error/ErrorCode.h>
#include <isce3/except/Error.h>
#include <isce3/cuda/focus/Backproject.h>
#include <isce3/cuda/signal/NFFT2d.h>
#include <isce3/focus/DryTroposphereModel.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <optional>
#include <pybind11/eigen.h>
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

    m.def("backproject_to_polar_grid", [](
                const py::array_t<std::complex<float>, py::array::c_style> in,
                const isce3::core::Linspace<double>& in_slant_range,
                const std::vector<isce3::core::Vec3>& pos,
                const std::vector<isce3::core::Vec3>& vel,
                const isce3::focus::PolarGrid& grid,
                const DEMInterpolator& dem,
                double fc,
                const Kernel<float>& kernel,
                const std::string& dry_tropo_model,
                py::dict rdr2geo_params) {

            if (in.ndim() != 2) {
                throw InvalidArgument(ISCE_SRCINFO(), "input signal data must be 2-D");
            }

            if (in.shape()[0] != pos.size() or
                in.shape()[1] != in_slant_range.size()) {

                std::string errmsg = "input signal data shape must match "
                    "input radar grid shape";
                throw InvalidArgument(ISCE_SRCINFO(), errmsg);
            }

            DryTroposphereModel atm = parseDryTropoModel(dry_tropo_model);

            const auto r2gparams = parse_rdr2geo_params(rdr2geo_params);

            const std::complex<float>* in_data = in.data();

            auto [err, outp, heightp] = [&]() {
                py::gil_scoped_release release;
                return isce3::cuda::focus::backprojectToPolarGrid(in_data,
                    in_slant_range, pos, vel, grid,
                    dem, fc, kernel, atm, r2gparams);
            }();

            // TODO bind ErrorCode class.  For now return nonzero on failure.
            bool status = err == ErrorCode::Success;

            auto out = move_to_numpy(std::move(outp),
                {static_cast<py::ssize_t>(grid.length()),
                 static_cast<py::ssize_t>(grid.width())});
            auto height = move_to_numpy(std::move(heightp),
                {static_cast<py::ssize_t>(grid.length()),
                 static_cast<py::ssize_t>(grid.width())});

            return std::make_tuple(status, out, height);
            },
            R"(
                Focus in azimuth via time-domain backprojection.
            )",
            py::arg("in"),
            py::arg("in_slant_range"),
            py::arg("position"),
            py::arg("velocity"),
            py::arg("out_grid"),
            py::arg("dem"),
            py::arg("fc"),
            py::arg("kernel"),
            py::arg("dry_tropo_model") = "nodelay",  // off here, on later
            py::arg("rdr2geo_params") = py::dict());

    m.def("project_polar_to_geo", [](
                py::array_t<std::complex<float>>& geo_image,
                const py::array_t<double>& geo_points,
                const isce3::focus::PolarGrid& grid,
                const py::array_t<std::complex<float>>& polar_image,
                const double wavelength,
                py::dict nfft2_params) {

            // get root finding parameters
            const auto params = parse_nfft2d_params(nfft2_params);
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

    m.def("accumulate_polar_images_to_radar_grid", [](
                py::array_t<std::complex<float>, py::array::c_style> out,
                const RadarGeometry& out_geometry,
                const isce3::core::Orbit& in_orbit,
                const isce3::core::LUT2d<double>& in_doppler,
                const std::vector<isce3::focus::PolarGrid>& grids,
                const py::sequence& py_image_interpolators,
                const DEMInterpolator& dem,
                double fc,
                double ds,
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

            const auto nimg = py_image_interpolators.size();
            if (grids.size() != nimg) {
                throw InvalidArgument(ISCE_SRCINFO(), "must have grid for each sub-image");
            }

            std::complex<float>* out_data = out.mutable_data();
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

            // Convert Python sequence to std::vector.  Element type is
            // pointer so this shouldn't involve any copy.
            using T = isce3::cuda::signal::NFFT2dResult<float>;
            auto interpolators = std::vector<const T*>(nimg);
            std::transform(py_image_interpolators.begin(),
                py_image_interpolators.end(), interpolators.begin(),
                [](const py::handle& py_itp) -> const T* {
                    return &(py_itp.cast<const T&>());
                });

            ErrorCode err;
            {
                py::gil_scoped_release release;
                err = isce3::cuda::focus::accumulatePolarImagesToRadarGrid(
                    out_data, out_geometry, in_orbit, in_doppler, grids,
                    interpolators, dem, fc, ds, atm, r2gparams, g2rparams,
                    height_data);
            }
            // TODO bind ErrorCode class.  For now return nonzero on failure.
            return err != ErrorCode::Success;
        },
        py::arg("out"),
        py::arg("out_geometry"),
        py::arg("in_orbit"),
        py::arg("in_doppler"),
        py::arg("grids"),
        py::arg("image_interpolators"),
        py::arg("dem"),
        py::arg("fc"),
        py::arg("ds"),
        py::arg("dry_tropo_model") = "tsx",
        py::arg("rdr2geo_params") = py::dict(),
        py::arg("geo2rdr_params") = py::dict(),
        py::arg("height") = py::none());

    m.def("merge_polar_images", [](
            const std::vector<isce3::focus::PolarGrid>& grids,
            py::sequence py_image_interpolators,
            const isce3::focus::PolarGrid& output_grid,
            Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> output_image,
            const double fc,
            const isce3::geometry::DEMInterpolator& dem,
            const py::dict rdr2geo_params,  // only difference for python
            int az_block_size)
        {
            // Convert Python sequence to std::vector.  Element type is
            // pointer so this shouldn't involve any copy.
            const auto nimg = py_image_interpolators.size();
            using T = isce3::cuda::signal::NFFT2dResult<float>;
            auto interpolators = std::vector<const T*>(nimg);
            std::transform(py_image_interpolators.begin(),
                py_image_interpolators.end(), interpolators.begin(),
                [](const py::handle& py_itp) -> const T* {
                    return &(py_itp.cast<const T&>());
                });

            const auto r2g_params = parse_rdr2geo_params(rdr2geo_params);

            return mergePolarImages(grids, interpolators, output_grid,
                output_image, fc, dem, r2g_params, az_block_size);
        },
        py::arg("grids"),
        py::arg("image_interpolators"),
        py::arg("output_grid"),
        py::arg("output_image"),
        py::arg("fc"),
        py::arg("dem") = DEMInterpolator(),
        py::arg("rdr2geo_parameters") = py::dict(),
        py::arg("az_block_size") = 1024
    );
}
