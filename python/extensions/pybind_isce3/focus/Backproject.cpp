#include "Backproject.h"

#include <algorithm>
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

NFFT2Params parse_nfft2_params(const py::dict& params)
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
                throw InvalidArgument(ISCE_SRCINFO(),
                    "unexpected NFFT keyword: " + key);
            }
        }
        return out;
    };
    NFFT2Params out;
    for (auto item : params) {
        auto key = item.first.cast<std::string>();
        if (key == "x") {
            out.x = parse_ms(item.second.cast<py::dict>());
        }
        else if (key == "y") {
            out.y = parse_ms(item.second.cast<py::dict>());
        }
        else {
            throw InvalidArgument(ISCE_SRCINFO(),
                "unexpected NFFT2Parms keyword: " + key);
        }
    }
    return out;
}

void addbinding(py::class_<PolarGrid>& pyPolarGrid)
{
    double aztime_start, aztime_end;
    isce3::core::Vec3 origin, axis;
    isce3::core::Linspace<double> range;
    isce3::core::Linspace<double> sin_squint;

    pyPolarGrid
        .def_readonly("aztime_start", &PolarGrid::aztime_start)
        .def_readonly("aztime_end", &PolarGrid::aztime_end)
        .def_readonly("origin", &PolarGrid::origin)
        .def_readonly("axis", &PolarGrid::axis)
        .def_readonly("range", &PolarGrid::range)
        .def_readonly("sin_squint", &PolarGrid::sin_squint)
        ;
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
                return isce3::focus::backprojectFirstStage(in_data,
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

    m.def("backproject_final_stage", [](
                py::array_t<std::complex<float>, py::array::c_style> out,
                const RadarGeometry& out_geometry,
                const isce3::core::Orbit& in_orbit,
                const isce3::core::LUT2d<double>& in_doppler,
                const std::vector<PolarGrid>& grids,
                const std::vector<py::array_t<std::complex<float>, py::array::c_style>>& images,
                const DEMInterpolator& dem,
                double fc,
                double ds,
                const Kernel<float>& kernel_rg,
                const Kernel<float>& kernel_az,
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

            if (grids.size() != images.size()) {
                throw InvalidArgument(ISCE_SRCINFO(), "must have grid for each sub-image");
            }
            for (decltype(grids.size()) i = 0; i < grids.size(); ++i) {
                const auto& grid = grids[i];
                const auto& image = images[i];
                if (image.ndim() != 2) {
                    throw InvalidArgument(ISCE_SRCINFO(), "input sub-images must be 2-D");
                }
                if (image.shape()[0] != grid.length() or
                        image.shape()[1] != grid.width()) {
                    std::string errmsg = "input sub-image shape must match "
                        "input radar grid shape";
                    throw InvalidArgument(ISCE_SRCINFO(), errmsg);
                }
            }

            std::vector<const std::complex<float>*> images_(images.size());
            std::transform(images.begin(), images.end(), images_.begin(),
                [](const auto& image) { return image.data(); });

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

            const auto r2gparams = parse_rdr2geo_params(rdr2geo_params);
            const auto g2rparams = parse_geo2rdr_params(geo2rdr_params);

            ErrorCode err;
            {
                py::gil_scoped_release release;
                err = backprojectFinalStage(out_data, out_geometry, in_orbit,
                    in_doppler, grids, images_,
                    dem, fc, ds, kernel_rg, kernel_az, r2gparams, g2rparams,
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
        py::arg("images"),
        py::arg("dem"),
        py::arg("fc"),
        py::arg("ds"),
        py::arg("kernel_rg"),
        py::arg("kernel_az"),
        py::arg("rdr2geo_params") = py::dict(),
        py::arg("geo2rdr_params") = py::dict(),
        py::arg("height") = py::none());

    m.def("find_polar_grid_bbox_in_radar_grid", [](
                const PolarGrid& polar_grid,
                const RadarGeometry& radar_geom,
                const DEMInterpolator& dem,
                py::dict rdr2geo_params,
                py::dict geo2rdr_params,
                int nextra) {

            const auto r2gparams = parse_rdr2geo_params(rdr2geo_params);
            const auto g2rparams = parse_geo2rdr_params(geo2rdr_params);

            auto [grid, status] = findPolarGridBoundingBoxInRadarGrid(polar_grid,
                radar_geom, dem, r2gparams, g2rparams, nextra);

            if (status != ErrorCode::Success) {
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                    "Could not determine polar grid bounds within radar grid.");
            }
            return grid;
        },
        py::arg("polar_grid"),
        py::arg("radar_geom"),
        py::arg("dem"),
        py::arg("rdr2geo_params") = py::dict(),
        py::arg("geo2rdr_params") = py::dict(),
        py::arg("nextra") = 0);

    m.def("computeRadarGridGeoPoints", [](
                const RadarGeometry& geom,
                const DEMInterpolator& dem,
                py::dict rdr2geo_params) {

            // get root finding parameters
            const auto r2gparams = parse_rdr2geo_params(rdr2geo_params);

            // allocate memory
            const py::ssize_t m = geom.gridLength(), n = geom.gridWidth();
            auto points = py::array_t<double, py::array::c_style>({m, n, 3L});

            // XXX type cast after checking sizes, assume alignment is okay
            // TODO redo with Eigen::Map or change interface from Vec3 to double[3]?
            using isce3::core::Vec3;
            static_assert(sizeof(Vec3) == (sizeof(double[3])));
            auto ptr = reinterpret_cast<Vec3*>(points.mutable_data());

            // run the thing
            auto status = computeRadarGridGeoPoints(ptr, geom, dem, r2gparams);

            if (status != ErrorCode::Success) {
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                    "Could not compute map projection of polar grid coords.");
            }
            return points;
        },
        py::arg("geom"),
        py::arg("dem"), 
        py::arg("rdr2geo_params") = py::dict());

    m.def("project_polar_to_geo", [](
                py::array_t<std::complex<float>>& geo_image,
                const py::array_t<double>& geo_points,
                const PolarGrid& grid,
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

            auto status = projectPolarToGeo(geo_image.mutable_data(), ptr,
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
