#include "Backproject.h"
#include "pybind_isce3/signal/NFFT2d.h"  // parse NFFT2d parameters

#include <algorithm>
#include <optional>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <isce3/container/RadarGeometry.h>
#include <isce3/core/Kernels.h>
#include <isce3/core/Linspace.h>
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
using isce3::signal::NFFT2dResult;


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
        // all properties are read-only, so instances are hashable
        .def_property_readonly("_members", [](const py::object self) {
            const auto q = py::getattr(self, "sin_squint").cast<Linspace<double>>();
            const auto r = py::getattr(self, "range").cast<Linspace<double>>();
            return py::make_tuple(
                py::getattr(self, "aztime_start"),
                py::getattr(self, "aztime_end"),
                py::tuple(py::getattr(self, "origin")),
                py::tuple(py::getattr(self, "axis")),
                py::make_tuple(q.first(), q.spacing(), q.size()),
                py::make_tuple(r.first(), r.spacing(), r.size()),
                py::getattr(self, "look_side")
            );
        })
        .def("__hash__", [](const py::object self) {
            return py::hash(py::getattr(self, "_members"));
        })
        .def("__eq__", [](const py::object self, const PolarGrid& typed_other) {
            // Strongly-typed function signature means we don't have to check
            // type.  The _members method is only defined in Python, though, so
            // cast to py::object.
            const py::object other = py::cast(typed_other);
            const auto a = getattr(self, "_members");
            const auto b = getattr(other, "_members");
            return a.equal(b);
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

                Parameters
                ----------
                out : numpy.ndarray[complex64]
                    Output 2D array of focused signal data.
                out_geometry : isce3.container.RadarGeometry
                    Target output grid, orbit, and Doppler.
                in : numpy.ndarray[complex64]
                    Input 2D array of range-compressed signal data.
                in_geometry : isce3.container.RadarGeometry
                    Input data grid, orbit, and Doppler.
                dem : isce3.geometry.DEMInterpolator
                    Digital elevation model.
                fc : float
                    Radar center frequency (Hz).
                ds : float
                    Desired azimuth resolution (m).
                kernel : isce3.core.Kernel
                    1-D interpolation kernel.
                dry_tropo_model : str, optional
                    Dry troposphere path delay model (defaults to "tsx").
                rdr2geo_params : dict, optional
                    rdr2geo_bracket configuration keyword arguments.
                geo2rdr_params : dict, optional
                    geo2rdr_bracket configuration keyword arguments.
                height : numpy.ndarray[float32], optional
                    Output array to store height of each pixel in meters above
                    the ellipsoid.

                Returns
                -------
                bool
                    True if successful, False if geometry fails to converge for
                    any pixel (those pixels are set to NaN).
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
        pri : float, optional
            Pulse repetition interval in s.  If variable, provide the PRI
            between the last pulse and the next one.  If not provided the
            average PRI will be used.

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
        py::arg("num_doppler_eval") = 2,
        py::arg("pri") = py::none());

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

            auto out = move_to_numpy(std::move(outp),
                {static_cast<py::ssize_t>(grid.length()),
                 static_cast<py::ssize_t>(grid.width())});
            auto height = move_to_numpy(std::move(heightp),
                {static_cast<py::ssize_t>(grid.length()),
                 static_cast<py::ssize_t>(grid.width())});

            return std::make_tuple(status, out, height);
            },
            R"(
                Focus in azimuth via time-domain backprojection onto a
                polar grid.

                Parameters
                ----------
                in : numpy.ndarray[complex64]
                    Input 2D array of range-compressed signal data.
                in_slant_range : isce3.core.Linspace
                    Slant range grid of the input data (m).
                position : list[numpy.ndarray]
                    Platform position vectors at each pulse (ECEF, m).
                velocity : list[numpy.ndarray]
                    Platform velocity vectors at each pulse (ECEF, m/s).
                out_grid : isce3.focus.PolarGrid
                    Target polar grid to backproject onto.
                dem : isce3.geometry.DEMInterpolator
                    Digital elevation model.
                fc : float
                    Radar center frequency (Hz).
                kernel : isce3.core.Kernel
                    1D interpolation kernel.
                dry_tropo_model : str, optional
                    Dry troposphere path delay model (defaults to "nodelay").
                rdr2geo_params : dict, optional
                    rdr2geo_bracket configuration keyword arguments.

                Returns
                -------
                tuple
                    - success : bool
                        True if successful, False if geometry fails
                        to converge for any pixel.
                    - out : numpy.ndarray[complex64]
                        Focused signal data on the polar grid.
                    - height : numpy.ndarray[float32]
                        Per-pixel height above the ellipsoid (m).
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

    m.def("merge_polar_grids", [](const std::vector<PolarGrid>& grids,
                                  const DEMInterpolator& dem,
                                  py::dict rdr2geo_params,
                                  const std::optional<double>& dq_min,
                                  const std::optional<double>& tq) {
            const auto r2g_params = parse_rdr2geo_params(rdr2geo_params);
            return mergePolarGrids(grids, dem, r2g_params, dq_min, tq);
        },
        R"(
            Create polar grid capable of sampling data from all input grids.

            Parameters
            ----------
            grids : list[isce3.focus.PolarGrid]
                List of subaperture grids.
            dem : isce3.geometry.DEMInterpolator, optional
                Digital elevation model reporting height (m) above the
                ellipsoid associated with its CRS.
            rdr2geo_params : dict, optional
                rdr2geo_bracket configuration keyword arguments.
            dq_min : float, optional
                Minimum allowed dimensionless Doppler spacing.
                Necessary for stripmap processing large subapertures.
            tq : float, optional
                Time constant for dimensionless Doppler spacing.
                If not provided it will be inferred from input grids.

            Returns
            -------
            isce3.focus.PolarGrid
                Merged output polar grid.
        )",
        py::arg("grids"),
        py::arg("dem") = DEMInterpolator(),
        py::arg("rdr2geo_params") = py::dict(),
        py::arg("dq_min") = py::none(),
        py::arg("tq") = py::none()
    );

    m.def("merge_polar_images", [](
            const std::vector<PolarGrid>& grids,
            const std::vector<NFFT2dResult<float>>& image_interpolators,  // copy :,(
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
        R"(
            Merge subaperture polar grid images into a single output grid.

            Combines multiple subaperture polar grid images onto a merged
            output polar grid. For each pixel in the output grid, the 3D
            target position is computed via polar2geo, and the input image
            data is accumulated via NFFT-based interpolation. The output
            image is expected to be zero-initialized by the caller.

            Parameters
            ----------
            grids : list[isce3.focus.PolarGrid]
                List of subaperture input polar grids.
            image_interpolators : list[numpy.ndarray]
                NFFT interpolators for each input grid.
            output_grid : isce3.focus.PolarGrid
                Merged output polar grid.
            output_image : numpy.ndarray[complex64]
                Accumulated output image (must be zero-initialized);
                dimensions must match output_grid.
            fc : float
                Center frequency (Hz).
            dem : isce3.geometry.DEMInterpolator, optional
                Digital elevation model.
            rdr2geo_parameters : dict, optional
                rdr2geo_bracket configuration keyword arguments.
            az_block_size : int, optional
                Number of azimuth rows to process at a time
                (defaults to 1024).

            Raises
            ------
            isce3.except.LengthError
                If output image dimensions or grid/interpolator
                counts are inconsistent.
            isce3.except.InvalidArgument
                If look directions are inconsistent or
                az_block_size is negative.
            isce3.except.DomainError
                If polar2geo fails to converge.
            isce3.except.RuntimeError
                If NFFT interpolation fails.
        )",
        py::arg("grids"),
        py::arg("image_interpolators"),
        py::arg("output_grid"),
        py::arg("output_image"),
        py::arg("fc"),
        py::arg("dem") = DEMInterpolator(),
        py::arg("rdr2geo_parameters") = py::dict(),
        py::arg("az_block_size") = 1024
    );

    m.def("accumulate_polar_images_to_radar_grid", [](
                py::array_t<std::complex<float>, py::array::c_style> out,
                const RadarGeometry& out_geometry,
                const isce3::core::Orbit& in_orbit,
                const isce3::core::LUT2d<double>& in_doppler,
                const std::vector<PolarGrid>& grids,
                const std::vector<NFFT2dResult<float>>& image_interpolators,
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

            DryTroposphereModel atm = parseDryTropoModel(dry_tropo_model);

            const auto r2gparams = parse_rdr2geo_params(rdr2geo_params);
            const auto g2rparams = parse_geo2rdr_params(geo2rdr_params);

            ErrorCode err;
            {
                py::gil_scoped_release release;
                err = accumulatePolarImagesToRadarGrid(out_data, out_geometry,
                    in_orbit, in_doppler, grids, image_interpolators, dem, fc,
                    ds, atm, r2gparams, g2rparams, height_data);
            }
            // TODO bind ErrorCode class.  For now return nonzero on failure.
            return err != ErrorCode::Success;
        },
        R"(
            Accumulate polar grid images onto an output stripmap radar grid.

            Combines multiple subaperture polar grid images together onto a
            stripmap radar geometry grid. For each pixel in the output grid,
            the target position is computed via rdr2geo, the corresponding
            coherent processing interval is determined via geo2rdr, and the
            polar image data is accumulated using NFFT-based interpolation.

            Parameters
            ----------
            out : numpy.ndarray[complex64]
                Output 2D array of focused signal data.
            out_geometry : isce3.container.RadarGeometry
                Target output grid, orbit, and Doppler.
            in_orbit : isce3.core.Orbit
                Input data orbit.
            in_doppler : isce3.core.LUT2d
                Input data Doppler centroid LUT.
            grids : list[isce3.focus.PolarGrid]
                List of subaperture polar grids.
            image_interpolators : list[numpy.ndarray]
                NFFT interpolators for each polar grid.
            dem : isce3.geometry.DEMInterpolator, optional
                Digital elevation model.
            fc : float
                Center frequency (Hz).
            ds : float
                Desired azimuth resolution (m).
            dry_tropo_model : str, optional
                Dry troposphere path delay model (defaults to "tsx").
            rdr2geo_params : dict, optional
                rdr2geo_bracket configuration keyword arguments.
            geo2rdr_params : dict, optional
                geo2rdr_bracket configuration keyword arguments.
            height : numpy.ndarray[float32], optional
                Output array to store height of each pixel in meters
                above the ellipsoid.

            Returns
            -------
            bool
                True if successful, False if rdr2geo or geo2rdr fails
                to converge for any pixel (those pixels are set to
                NaN).
        )",
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

    m.def("find_polar_grid_bbox_in_radar_grid", [](
                const PolarGrid& polar_grid,
                const RadarGeometry& radar_geom,
                const DEMInterpolator& dem,
                py::dict rdr2geo_params,
                py::dict geo2rdr_params,
                int nextra) {

            const auto r2gparams = parse_rdr2geo_params(rdr2geo_params);
            const auto g2rparams = parse_geo2rdr_params(geo2rdr_params);

            auto [i0, i1, j0, j1, status] = findPolarGridBoundingBoxInRadarGrid(
                polar_grid, radar_geom, dem, r2gparams, g2rparams, nextra);

            if (status != ErrorCode::Success) {
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                    "Could not determine polar grid bounds within radar grid.");
            }
            auto rows = py::slice(
                static_cast<py::ssize_t>(i0),
                static_cast<py::ssize_t>(i1),
                std::nullopt);
            auto cols = py::slice(
                static_cast<py::ssize_t>(j0),
                static_cast<py::ssize_t>(j1),
                std::nullopt);
            return std::make_tuple(rows, cols);
        },
        R"(
            Find the subset of a radar grid covered by a polar grid.

            Computes the bounding box of the polar grid in stripmap radar
            coordinates, then converts this bounding box to integer radar
            grid indices (azimuth line, range sample). If the polar grid
            does not overlap the radar grid at all, a zero-sized subset
            (0, 0, 0, 0) is returned.

            Parameters
            ----------
            polar_grid : isce3.focus.PolarGrid
                Input polar grid.
            radar_geom : isce3.container.RadarGeometry
                Target radar geometry grid.
            dem : isce3.geometry.DEMInterpolator
                Digital elevation model.
            rdr2geo_params : dict, optional
                rdr2geo_bracket configuration keyword arguments.
            geo2rdr_params : dict, optional
                geo2rdr_bracket configuration keyword arguments.
            nextra : int, optional
                Number of extra perimeter points per edge (defaults to 0).

            Returns
            -------
            tuple[slice, slice]
                - rows : slice
                    Azimuth line range (start, end).
                - cols : slice
                    Range sample range (start, end).

            Raises
            ------
            isce3.except.RuntimeError
                If the polar grid bounds cannot be determined.
        )",
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
        R"(
            Compute 3D geo coordinates for a radar grid.

            Computes the 3D XYZ position for every pixel in a radar
            geometry grid using rdr2geo_bracket.

            Parameters
            ----------
            geom : isce3.container.RadarGeometry
                Radar geometry grid, orbit, and Doppler.
            dem : isce3.geometry.DEMInterpolator
                Digital elevation model.
            rdr2geo_params : dict, optional
                rdr2geo_bracket configuration keyword arguments.

            Returns
            -------
            numpy.ndarray[float64]
                3D XYZ positions (ECEF, m) with shape (m, n, 3), where m is the
                grid length and n is the grid width.

            Raises
            ------
            isce3.except.RuntimeError
                If rdr2geo fails to converge for any pixel.
        )",
        py::arg("geom"),
        py::arg("dem"),
        py::arg("rdr2geo_params") = py::dict());

    m.def("accumulate_polar_image_to_geo_points", [](
                py::array_t<std::complex<float>, py::array::c_style>& image,
                const py::array_t<double, py::array::c_style>& xyz,
                const PolarGrid& grid,
                const NFFT2dResult<float>& nfft,
                const double wavelength,
                const std::optional<py::array_t<bool, py::array::c_style>>& mask) {

            const auto n = image.size();
            if (xyz.size() != 3 * n) {
                throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "shape mismatch between geo image and position arrays");
            }
            if (xyz.shape(xyz.ndim() - 1) != 3) {
                throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "expected trailing dimension size == 3 for XYZ points");
            }
            if (mask.has_value() and (mask.value().size() != n)) {
                throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "pixel mask size does not equal image size");
            }

            // XXX type cast after checking sizes
            using isce3::core::Vec3;
            static_assert(sizeof(Vec3) == (sizeof(double[3])));
            const auto ptr = reinterpret_cast<const Vec3*>(xyz.data());

            std::optional<const bool*> mask_ptr = std::nullopt;
            if (mask.has_value()) {
                mask_ptr = mask.value().data();
            }

            auto status = accumulatePolarImageToGeoPoints(
                image.mutable_data(), ptr, n, grid, nfft, wavelength, mask_ptr);

            if (status != ErrorCode::Success) {
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                    "Could not compute map projection of polar grid coords.");
            }
        },
        R"(
            Interpolate a polar grid image to given XYZ positions.

            Accumulates (adds) contributions from a polar grid image into
            an output complex signal array at specified 3D positions. For
            each position, the target location in the polar grid is
            computed via geo2polar, and the image is interpolated using
            NFFT. The phase is compensated by the wavenumber-range
            product kw * range.

            Parameters
            ----------
            image : numpy.ndarray[complex64]
                Output complex signal data (accumulates, so caller
                must init to zero).
            xyz : numpy.ndarray[float64]
                Target 3D positions (ECEF, m) with shape (n, 3).
            grid : isce3.focus.PolarGrid
                Polar grid containing the image data.
            nfft : numpy.ndarray
                NFFT interpolator for the polar grid.
            wavelength : float
                Radar wavelength (m).
            mask : numpy.ndarray[bool], optional
                Pixel mask; pixels with false are skipped.

            Raises
            ------
            isce3.except.LengthError
                If shape mismatch between geo image and position
                arrays, or if mask size does not equal image size.
            isce3.except.RuntimeError
                If the computation fails.
        )",
        py::arg("image"),
        py::arg("xyz"),
        py::arg("grid"),
        py::arg("nfft"),
        py::arg("wavelength"),
        py::arg("mask") = py::none());
}
