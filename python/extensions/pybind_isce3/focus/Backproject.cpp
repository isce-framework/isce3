#include "Backproject.h"
#include "pybind_isce3/signal/NFFT2d.h"  // parse NFFT2d parameters

#include <algorithm>
#include <optional>
#include <pybind11/eigen.h>
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
#include <isce3/signal/NFFT2d.h>

namespace py = pybind11;

using namespace isce3::focus;

using isce3::container::RadarGeometry;
using isce3::core::Kernel;
using isce3::core::EArray2D;
using isce3::error::ErrorCode;
using isce3::except::InvalidArgument;
using isce3::geometry::DEMInterpolator;
using isce3::geometry::detail::Rdr2GeoBracketParams;
using isce3::geometry::detail::Geo2RdrBracketParams;
using isce3::signal::NFFT2d;


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


void addbinding(py::class_<PolarGrid>& pyPolarGrid)
{
    using isce3::core::Vec3;
    using isce3::core::Linspace;
    using isce3::core::LookSide;

    pyPolarGrid
        .def(py::init<double, double, Vec3, Vec3, Linspace<double>, Linspace<double>, LookSide>(),
            py::arg("aztime_start"),
            py::arg("aztime_end"),
            py::arg("origin"),
            py::arg("axis"),
            py::arg("range"),
            py::arg("sin_squint"),
            py::arg("look_side")
        )
        .def_readonly("aztime_start", &PolarGrid::aztime_start)
        .def_readonly("aztime_end", &PolarGrid::aztime_end)
        .def_readonly("origin", &PolarGrid::origin)
        .def_readonly("axis", &PolarGrid::axis)
        .def_readonly("range", &PolarGrid::range)
        .def_readonly("sin_squint", &PolarGrid::sin_squint)
        .def_readonly("look_side", &PolarGrid::look_side)
        .def_property_readonly("shape", [](const PolarGrid& self) {
            return std::make_tuple(self.sin_squint.size(), self.range.size());
        })
        .def("__repr__", [](const py::object self) {
            std::vector<std::string> keys {"aztime_start", "aztime_end",
                    "origin", "axis", "range", "sin_squint", "look_side"};
            std::string out("PolarGrid(");
            for (auto it = keys.begin(); it != keys.end(); ++it) {
                    auto key = *it;
                    auto ckey = key.c_str();
                    out += key + "=" + std::string(py::str(self.attr(ckey)));
                    if (it != keys.end() - 1)
                            out += ", ";
            }
            return out + ")";
        })
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

    m.def("setup_polar_grid_for_pulses", &setupPolarGridForPulses,
        R"(
        Setup a polar (range-Doppler) grid corresponding to a set of pulses.

        Parameters
        ----------
        in_geometry : isce3.container.RadarGeometry
        azimuth_time : Sequence[float]
        range_bandwidth : float
        azimuth_resolution : float
        oversample_range : float, optional
        oversample_azimuth : float, optional
        num_doppler_eval : int, optional
            Number of points across swath to evaluate Doppler centroid to
            bound the variation of the centroid.  Default = 2
        densify_for_fast_transform : bool, optional
            Whether to increase the sample rate to achieve grid dimensions that
            are products of small prime factors (good for FFTs).
            Default = False

        Returns
        -------
        polar_grid : isce3.focus.PolarGrid
            Polar grid that efficiently samples the raw data.
        position : list[numpy.ndarray]
            Sensor position at each input pulse time.
        velocity : list[numpy.ndarray]
            Sensor velocity at each input pulse time.
        )",
        py::arg("in_geometry"),
        py::arg("azimuth_time"),
        py::arg("range_bandwidth"),
        py::arg("azimuth_resolution"),
        py::arg("oversample_range") = 1.2,
        py::arg("oversample_azimuth") = 1.2,
        py::arg("num_doppler_eval") = 2);

    m.def("get_polar_angle_time_constant", &getPolarAngleTimeConstant,
        R"(
        Get the time constant associated with polar angle spacing

        Parameters
        ----------
        fc : float
           Radar center frequency, Hz
        vs : float
           Satellite velocity (along azimuth axis), m/s
        bandwidth : float, optional
           Radar bandwidth, Hz (defaults to zero, e.g., narrow band)
        c : float, optional
           Speed of light, m/s (defaults to vacuum sol)

        Returns
        -------
        tq : float
           Time constant $T_q$, s

        This time constant is used to determine the sampling requirement for the
        sine of the squint angle (dimensionless Doppler)
             $$ q = \frac{\vec{v}}{v} \cdot \hat{l} $$
        where $\vec{v}$ is the velocity and $\hat{l}$ is the line-of-sight direction.
        Specifically, the Nyquist criterion is
             $$ \Delta q \leq \frac{T_q}{T_{sa}} $$
        where $T_{sa}$ is the time duration of the synthetic aperture.

        Helps implement equation (11) in @cite yegulalp2013
        )",
        py::arg("fc"),
        py::arg("vs"),
        py::arg("bandwidth") = 0.0,
        py::arg("c") = isce3::core::speed_of_light);

    m.def("backproject_to_polar_grid", [](
                const py::array_t<std::complex<float>, py::array::c_style> in,
                const isce3::core::Linspace<double>& in_slant_range,
                const std::vector<isce3::core::Vec3>& pos,
                const std::vector<isce3::core::Vec3>& vel,
                const PolarGrid& grid,
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

            if (pos.size() != vel.size()) {
                throw InvalidArgument(ISCE_SRCINFO(), "must provide same "
                    "number of position and velocity vectors");
            }

            DryTroposphereModel atm = parseDryTropoModel(dry_tropo_model);

            const auto r2gparams = parse_rdr2geo_params(rdr2geo_params);

            const std::complex<float>* in_data = in.data();

            auto [err, outp, heightp] = [&]() {
                py::gil_scoped_release release;
                return isce3::focus::backprojectToPolarGrid(in_data,
                    in_slant_range, pos, vel, grid,
                    dem, fc, kernel, atm, r2gparams);
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
            py::arg("in_slant_range"),
            py::arg("position"),
            py::arg("velocity"),
            py::arg("out_grid"),
            py::arg("dem"),
            py::arg("fc"),
            py::arg("kernel"),
            py::arg("dry_tropo_model") = "tsx",
            py::arg("rdr2geo_params") = py::dict());

    m.def("merge_polar_grids", [](const std::vector<PolarGrid>& grids,
                                  const DEMInterpolator& dem,
                                  py::dict rdr2geo_params,
                                  const std::optional<double>& dq_min,
                                  const std::optional<double>& tq) {
            const auto r2g_params = parse_rdr2geo_params(rdr2geo_params);
            return mergePolarGrids(grids, dem, r2g_params, dq_min, tq);
        },
        py::arg("grids"),
        py::arg("dem") = DEMInterpolator(),
        py::arg("rdr2geo_params") = py::dict(),
        py::arg("dq_min") = py::none(),
        py::arg("tq") = py::none()
    );

    m.def("merge_polar_images", [](
            const std::vector<PolarGrid>& grids,
            const std::vector<NFFT2d<float>>& image_interpolators,  // copy :,(
            const PolarGrid& output_grid,
            Eigen::Ref<EArray2D<std::complex<float>>> output_image,
            const double fc,
            const isce3::geometry::DEMInterpolator& dem,
            const py::dict rdr2geo_params,  // only difference for python
            int az_block_size)
        {
            const auto r2g_params = parse_rdr2geo_params(rdr2geo_params);
            return mergePolarImages(grids, image_interpolators, output_grid,
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

    m.def("backproject_final_stage", [](
                py::array_t<std::complex<float>, py::array::c_style> out,
                const RadarGeometry& out_geometry,
                const isce3::core::Orbit& in_orbit,
                const isce3::core::LUT2d<double>& in_doppler,
                const std::vector<PolarGrid>& grids,
                const std::vector<NFFT2d<float>>& image_interpolators,
                const DEMInterpolator& dem,
                double fc,
                double ds,
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

            if (grids.size() != image_interpolators.size()) {
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

            const auto r2gparams = parse_rdr2geo_params(rdr2geo_params);
            const auto g2rparams = parse_geo2rdr_params(geo2rdr_params);

            ErrorCode err;
            {
                py::gil_scoped_release release;
                err = backprojectFinalStage(out_data, out_geometry, in_orbit,
                    in_doppler, grids, image_interpolators, dem, fc, ds,
                    r2gparams, g2rparams, height_data);
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
                const NFFT2d<float>& nfft,
                const double wavelength) {

            // get root finding parameters
            if (geo_points.size() != 3 * geo_image.size()) {
                throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "shape mismatch between geo image and position arrays");
            }
            auto n = static_cast<size_t>(geo_image.size());

            // XXX type cast after checking sizes, assume alignment is okay
            // TODO redo with Eigen::Map or change interface from Vec3 to double[3]?
            using isce3::core::Vec3;
            static_assert(sizeof(Vec3) == (sizeof(double[3])));
            const auto ptr = reinterpret_cast<const Vec3*>(geo_points.data());

            auto status = projectPolarToGeo(geo_image.mutable_data(), ptr,
                n, grid, nfft, wavelength);

            if (status != ErrorCode::Success) {
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                    "Could not compute map projection of polar grid coords.");
            }
        },
        py::arg("geo_image"),
        py::arg("geo_points"),
        py::arg("grid"),
        py::arg("nfft"),
        py::arg("wavelength"));
}
