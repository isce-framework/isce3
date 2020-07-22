#include "boundingbox.h"

#include <cstdlib>
#include <ogr_geometry.h>
#include <string>

#include <isce3/core/Orbit.h>
#include <isce3/core/Projections.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/boundingbox.h>
#include <isce3/product/RadarGridParameters.h>

using namespace isce3::core;
using namespace isce3::geometry;
using isce3::product::RadarGridParameters;
namespace py = pybind11;

void addbinding_boundingbox(py::module& m)
{
    // TODO actually bind the C++ functions.  For now avoid wrapping the GDAL
    // output types by defining a new function that just returns the WKT string.
    m.def(
            "get_geo_perimeter_wkt",
            [](const RadarGridParameters& grid, const Orbit& orbit,
               const LUT2d<double>& doppler, const DEMInterpolator& dem,
               int pointsPerEdge, double threshold, int numiter) {
                auto proj = LonLat();
                auto perimeter =
                        getGeoPerimeter(grid, orbit, &proj, doppler, dem,
                                        pointsPerEdge, threshold, numiter);
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
            py::arg("points_per_edge") = 11, py::arg("threshold") = 1e-8,
            py::arg("numiter") = 15, R"(
    Compute the perimeter of a radar grid in map coordinates.

    The output of this method is the WKT representation of an OGRLinearRing.
    The sequence of walking the perimeter is always in the following order:

    Start at Early Time, Near Range edge. Always the first point of the polygon.
    From there, Walk along the Early Time edge to Early Time, Far Range.
    From there, walk along the Far Range edge to Late Time, Far Range.
    From there, walk along the Late Time edge to Late Time, Near Range.
    From there, walk along the Near Range edge back to Early Time, Near Range.
    )");
}
