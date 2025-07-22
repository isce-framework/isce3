#include "boundingbox.h"

#include <cstdlib>
#include <ogr_geometry.h>
#include <string>
#include <pybind11/stl.h>

#include <isce3/core/Orbit.h>
#include <isce3/core/Projections.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/detail/Rdr2Geo.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/boundingbox.h>
#include <isce3/product/RadarGridParameters.h>
#include <isce3/product/GeoGridParameters.h>
using namespace isce3::core;
using namespace isce3::geometry;
using isce3::product::RadarGridParameters;
using isce3::product::GeoGridParameters;
namespace py = pybind11;

void addbinding(py::class_<RadarGridBoundingBox> &pyRadarBoundingBox)
{
    pyRadarBoundingBox
        .def(py::init<const int, const int, const int, const int>(),
            py::arg("first_azimuth_line"),
            py::arg("last_azimuth_line"),
            py::arg("first_range_sample"),
            py::arg("last_range_sample"))
        .def_readwrite("first_azimuth_line",
                &RadarGridBoundingBox::firstAzimuthLine)
        .def_readwrite("last_azimuth_line",
                &RadarGridBoundingBox::lastAzimuthLine)
        .def_readwrite("first_range_sample",
                &RadarGridBoundingBox::firstRangeSample)
        .def_readwrite("last_range_sample",
                &RadarGridBoundingBox::lastRangeSample)
        ;
}

void addbinding_boundingbox(py::module& m)
{
    const isce3::geometry::detail::Geo2RdrParams defaults;
    // TODO actually bind the C++ functions.  For now avoid wrapping the GDAL
    // output types by defining a new function that just returns the WKT string.
    m.def(
            "get_geo_perimeter_wkt",
            [](const RadarGridParameters& grid, const Orbit& orbit,
               const LUT2d<double>& doppler, const DEMInterpolator& dem,
               int pointsPerEdge, double threshold) {
                auto proj = LonLat();
                auto perimeter =
                        getGeoPerimeter(grid, orbit, &proj, doppler, dem,
                                        pointsPerEdge, threshold);
                // convert ring to polygon
                auto poly = OGRPolygon();
                poly.addRing(&perimeter);
                char* c_wkt = NULL;
                poly.exportToWkt(&c_wkt);
                // Not sure if pybind11 frees a char*, so wrap in std::string.
                auto wkt = std::string(c_wkt);
                std::free(c_wkt);
                return wkt;
            },
            py::arg("grid"), py::arg("orbit"),
            py::arg("doppler") = LUT2d<double>(),
            py::arg("dem") = DEMInterpolator(0.),
            py::arg("points_per_edge") = 11,
            py::arg("threshold") = isce3::geometry::detail::DEFAULT_TOL_HEIGHT,
    R"(
    Compute the perimeter of a radar grid in map coordinates.

    The output of this method is the WKT representation of an OGRLinearRing.
    The sequence of walking the perimeter is always in the following order:

    Start at Early Time, Near Range edge. Always the first point of the polygon.
    From there, Walk along the Early Time edge to Early Time, Far Range.
    From there, walk along the Far Range edge to Late Time, Far Range.
    From there, walk along the Late Time edge to Late Time, Near Range.
    From there, walk along the Near Range edge back to Early Time, Near Range.
    )")
    .def("get_radar_bbox",
            py::overload_cast<const isce3::product::GeoGridParameters&,
                const isce3::product::RadarGridParameters&,
                const isce3::core::Orbit&, const float,
                const float, const isce3::core::LUT2d<double>&, const int,
                const isce3::geometry::detail::Geo2RdrParams&, const int>
                (&isce3::geometry::getRadarBoundingBox),
            py::arg("geo_grid"),
            py::arg("radar_grid"),
            py::arg("orbit"),
            py::arg("min_height") = isce3::core::GLOBAL_MIN_HEIGHT,
            py::arg("max_height") = isce3::core::GLOBAL_MAX_HEIGHT,
            py::arg("doppler") = LUT2d<double>(),
            py::arg("margin") = 50,
            py::arg("geo2rdr_params") = defaults,
            py::arg("geogrid_expansion_threshold") = 100,
            R"(
    Compute the bounding box of a geocoded grid in the radar coordinates. An
    exception is raised if any corners fails to convergers or any computed
    bounding box index is overlaps or is out of bounds.

    Parameters
    ----------
    geoGrid: GeoGridParameters
        Geo grid whose radar grid bounding box indices are to be computed
    radarGrid: RadarGridParameters
        Radar grid that computed indices are computed with respect to
    orbit: Orbit
        Orbit object
    min_height: float
        Minimum height values used in geo2rdr computations
    max_height: float
        Maximum height values used in geo2rdr computations
    doppler: LUT2d
        LUT2d doppler model of the radar image grid
    margin: int
        Margin to add to estimated bounding box in  pixels (default 5)
    geo2rdr_params: isce3.geometry.detail.geo2rdr_params
        Structure containing the following geo2rdr parameters:
        Azimuth time threshold for convergence (default 1e-8 sec)
        Max number of iterations for convergence (default: 50)
        Step size used for computing derivative of doppler (default: 10 m)
    geogrid_expansion_threshold: int
        Number of geogrid expansions if geo2rdr fails (default: 100)

    Returns
    -------
    RadarGridBoundingBox
        Radar grid bounding box object with indices for:
        first_azimuth_line,last_azimuth_line, first_range_sample,
        last_range_sample
    )")
    .def("get_radar_bbox",
            py::overload_cast<const isce3::product::GeoGridParameters&,
                const isce3::product::RadarGridParameters&,
                const isce3::core::Orbit&, isce3::geometry::DEMInterpolator&,
                const isce3::core::LUT2d<double>&, const int,
                const isce3::geometry::detail::Geo2RdrParams&, const int>
                (&isce3::geometry::getRadarBoundingBox),
            py::arg("geo_grid"),
            py::arg("radar_grid"),
            py::arg("orbit"),
            py::arg("dem") = DEMInterpolator(0.),
            py::arg("doppler") = LUT2d<double>(),
            py::arg("margin") = 5,
            py::arg("geo2rdr_params") = defaults,
            py::arg("geogrid_expansion_threshold") = 100,
            R"(
    Compute the bounding box of a geocoded grid in the radar coordinates. An
    exception is raised if any corners fails to convergers or any computed
    bounding box index is overlaps or is out of bounds.

    Parameters
    ----------
    geoGrid: GeoGridParameters
        Geo grid whose radar grid bounding box indices are to be computed
    radarGrid: RadarGridParameters
        Radar grid that computed indices are computed with respect to
    orbit: Orbit
        Orbit object
    demInterp: DEMInterpolator
        DEM interpolator object used compute min and max height values used in
        geo2rdr computations
    doppler: LUT2d
        LUT2d doppler model of the radar image grid
    margin: int
        Margin to add to estimated bounding box in  pixels (default 5)
    geo2rdr_params: isce3.geometry.detail.geo2rdr_params
        Structure containing the following geo2rdr parameters:
        Azimuth time threshold for convergence (default 1e-8 sec)
        Max number of iterations for convergence (default: 50)
        Step size used for computing derivative of doppler (default: 10 m)
    geogrid_expansion_threshold: int
        Number of geogrid expansions if geo2rdr fails (default: 100)

    Returns
    -------
    _: RadarGridBoundingBox
        Radar grid bounding box object with indices for:
        first_azimuth_line,last_azimuth_line, first_range_sample,
        last_range_sample
    )");
}
