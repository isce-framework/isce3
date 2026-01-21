#include "DEMInterpolator.h"

#include <memory>
#include <pybind11/eigen.h>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Projections.h>
#include <isce3/geometry/loadDem.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>

#include <pybind_isce3/core/Constants.h>

namespace py = pybind11;

using DEMInterp = isce3::geometry::DEMInterpolator;

void addbinding(pybind11::class_<DEMInterp>& pyDEMInterpolator)
{
    pyDEMInterpolator
            .def(py::init<double, isce3::core::dataInterpMethod, int>(),
                    py::arg("height") = 0.0,
                    py::arg("method") = isce3::core::BILINEAR_METHOD,
                    py::arg("epsg") = 4326)
            // For convenience allow a string, too.
            .def(py::init([](double h, const std::string& method, int epsg) {
                auto m = parseDataInterpMethod(method);
                return DEMInterp(h, m, epsg);
            }),
                    py::arg("height") = 0.0, py::arg("method") = "bilinear",
                    py::arg("epsg") = 4326)

            // This constructor is similar to method "loadDEM" but is more
            // convenient!
            .def(py::init([](isce3::io::Raster& raster_obj) {
                DEMInterp dem {};
                dem.loadDEM(raster_obj);
                return dem;
            }),
                    "Construct DEM from ISCE3 Raster object",
                    py::arg("raster_obj"))

            .def("load_dem",
                    py::overload_cast<isce3::io::Raster&, int>
                    (&DEMInterp::loadDEM),
                    py::arg("raster"), py::arg("raster_band") = 1)
            .def("load_dem",
                    py::overload_cast<isce3::io::Raster&, double, double,
                            double, double, int>(&DEMInterp::loadDEM),
                    py::arg("raster"), py::arg("min_x"), py::arg("max_x"),
                    py::arg("min_y"), py::arg("max_y"), py::arg("raster_band") = 1)

            .def("interpolate_lonlat", &DEMInterp::interpolateLonLat,
                R"(
                Interpolate the digital elevation model at a given point.

                Parameters
                ----------
                longitude : float
                    Longitude in radians
                latitude : float
                    Geodetic latitude in radians

                Returns
                -------
                height : float
                    DEM value interpolated at requested point, or ref_height
                    if point is not within the spatial extent of the DEM.
                    This class does not perform any conversions for units or
                    vertical datum, though many places in ISCE3 assume the value
                    represents the height in meters above the ellipsoid
                    associated with the DEM projection.
                )",
                py::arg("longitude"),
                py::arg("latitude")
            )
            .def("interpolate_xy", &DEMInterp::interpolateXY)

            .def_property("ref_height",
                    py::overload_cast<>(&DEMInterp::refHeight, py::const_),
                    py::overload_cast<double>(&DEMInterp::refHeight))
            .def_property_readonly("have_raster", &DEMInterp::haveRaster)
            .def_property_readonly("have_stats", &DEMInterp::haveStats)
            .def_property("interp_method",
                    py::overload_cast<>(&DEMInterp::interpMethod, py::const_),
                    [](DEMInterp& self, py::object method) {
                        self.interpMethod(duck_method(method));
                    })

            .def("compute_min_max_mean_height",
                    [](DEMInterp& self) {
                        float dem_min, dem_max, dem_avg;
                        self.computeMinMaxMeanHeight(dem_min, dem_max,
                                                     dem_avg);
                    })
            .def_property_readonly("mean_height", &DEMInterp::meanHeight)
            .def_property_readonly("min_height", &DEMInterp::minHeight)
            .def_property_readonly("max_height", &DEMInterp::maxHeight)

            // Define all these as readonly even though writable in C++ API.
            // Probably better to just convert your data to a GDAL format than
            // try to build a DEM on the fly.
            .def_property_readonly(
                    "data",
                    [](DEMInterp& self) { // .data() isn't const
                        if (!self.haveRaster()) {
                            throw std::out_of_range(
                                    "Tried to access DEM data but size=0");
                        }
                        using namespace Eigen;
                        using MatF = Eigen::Matrix<float, Dynamic, Dynamic,
                                RowMajor>;
                        Map<const MatF> mat(
                                self.data(), self.length(), self.width());
                        return mat;
                    },
                    py::return_value_policy::reference_internal)
            .def_property_readonly("x_start",
                    py::overload_cast<>(&DEMInterp::xStart, py::const_))
            .def_property_readonly("y_start",
                    py::overload_cast<>(&DEMInterp::yStart, py::const_))
            .def_property_readonly("delta_x",
                    py::overload_cast<>(&DEMInterp::deltaX, py::const_))
            .def_property_readonly("delta_y",
                    py::overload_cast<>(&DEMInterp::deltaY, py::const_))
            .def_property_readonly(
                    "width", py::overload_cast<>(&DEMInterp::width, py::const_))
            .def_property_readonly("length",
                    py::overload_cast<>(&DEMInterp::length, py::const_))
            .def_property_readonly("epsg_code",
                    py::overload_cast<>(&DEMInterp::epsgCode, py::const_))
            .def_property_readonly("mid_lon_lat", &DEMInterp::midLonLat)
            .def_property_readonly("ellipsoid", [](DEMInterp& self) {
                isce3::core::Ellipsoid ellipsoid =
                    makeProjection(self.epsgCode())->ellipsoid();
                return ellipsoid;
            });
}

void addbinding_DEM_raster2interpolator(py::module& m)
{
    m.def("dem_raster_to_interpolator",
        py::overload_cast<isce3::io::Raster &,
            const isce3::product::GeoGridParameters &, const int,
            const isce3::core::dataInterpMethod>
            (&isce3::geometry::DEMRasterToInterpolator),
        py::arg("dem_raster"),
        py::arg("geo_grid"),
        py::arg("dem_margin_in_pixels") = 50,
        py::arg("dem_interp_method") = isce3::core::BIQUINTIC_METHOD,
        R"(
    Returns a DEM interpolator for a geocoded grid.

    The geocoded grid and the input raster of the DEM can be in different or
    same projection systems

    Parameters
    ----------
    dem_raster: isce3.io.Raster
        Raster of the DEM
    geo_grid: isce3.product.GeoGridParameters
        Parameters of the geocoded grid
    dem_margin_in_pixels: int
        DEM extra margin in pixels
    dem_interp_method: isce3.core.DataInterpMethod
        DEM interpolation method

    Returns
    -------
    _: isce3.geometry.DEMInterpolator
        DEM interpolator for given DEM raster and geo grid.
        )")
    .def("dem_raster_to_interpolator",
        py::overload_cast<isce3::io::Raster &,
            const isce3::product::GeoGridParameters &,
            const int, const int, const int, const int,
            const isce3::core::dataInterpMethod>
            (&isce3::geometry::DEMRasterToInterpolator),
        py::arg("dem_raster"),
        py::arg("geo_grid"),
        py::arg("line_start"),
        py::arg("block_length"),
        py::arg("block_width"),
        py::arg("dem_margin_in_pixels") = 50,
        py::arg("dem_interp_method") = isce3::core::BIQUINTIC_METHOD,
        R"(
    Returns a DEM interpolator for a block within a geocoded grid.

    The geocoded grid and the input raster of the DEM can be in different or
    same projection systems

    Parameters
    ----------
    dem_raster: isce3.io.Raster
        Raster of the DEM
    geo_grid: isce3.product.GeoGridParameters
        Parameters of the geocoded grid
    line_start: int
        Starting line of block of interest in geocoded grid
    block_length: int
        Length of block of interest in geocoded grid
    block_width: int
        Width of block of interest in geocoded grid
    dem_margin_in_pixels: int
        DEM extra margin in pixels
    dem_interp_method: isce3.core.DataInterpMethod
        DEM interpolation method

    Returns
    -------
    _: isce3.geometry.DEMInterpolator
        DEM interpolator for given DEM raster and geo grid.
        )")
    ;
}


void addbinding_load_dem_from_proj(py::module& m)
{
    m.def("load_dem_from_proj",
        [](isce3::io::Raster &dem_raster,
            const double x0,
            const double xf,
            const double minY,
            const double maxY,
            const isce3::core::dataInterpMethod dem_interp_method,
            isce3::core::ProjectionBase* proj,
            const int dem_margin_x_in_pixels,
            const int dem_margin_y_in_pixels,
            const int dem_raster_band) {

            DEMInterp dem_interp(0, dem_interp_method);

            isce3::geometry::loadDemFromProj(dem_raster,
                                             x0,
                                             xf,
                                             minY,
                                             maxY,
                                             &dem_interp,
                                             proj,
                                             dem_margin_x_in_pixels,
                                             dem_margin_y_in_pixels,
                                             dem_raster_band);

            return dem_interp;
        },
        py::arg("dem_raster"),
        py::arg("x0"),
        py::arg("xf"),
        py::arg("min_y"),
        py::arg("max_y"),
        py::arg("dem_interp_method") = isce3::core::BIQUINTIC_METHOD,
        py::arg("proj") = nullptr,
        py::arg("dem_margin_x_in_pixels") = 100,
        py::arg("dem_margin_y_in_pixels") = 200,
        py::arg("dem_raster_band") = 1,
        R"(
    Load DEM raster into a DEMInterpolator object around a given bounding box
    in the same or different coordinate system as the DEM raster

    Parameters
    ----------
    dem_raster: isce3.io.Raster
        Raster of the DEM
    x0: double
        If the DEM is in geographic coordinates and the `x0` coordinate is not
        from the polar stereo system EPSG 3031 or EPSG 3413, this point represents
        the minimum X coordinate value. In this case, the maximum
        longitude span that this function can handle is 180 degrees
        (when the DEM is in geographic coordinates and `proj` is in polar stereo        
    xf: double
        Easting/longitude of eastern edge of bounding box
        If the DEM is in geographic coordinates and the `xf` coordinate is not
        from the polar stereo system EPSG 3031 or EPSG 3413, this point represents
        the maximum X coordinate value. In this case, the maximum
        longitude span that this function can handle is 180 degrees
        (when the DEM is in geographic coordinates and `proj` is in polar stereo)
    min_y: double
        Minimum Y/northing position
    max_y: double
        Maximum Y/northing position
    dem_interp_method: isce3.core.DataInterpMethod
        DEM interpolation method
    proj: 
        Projection object (nullptr to use same DEM projection)
    dem_margin_x_in_pixels, int
        DEM X/easting margin in pixels
    dem_margin_y_in_pixels, int
        DEM Y/northing margin in pixels
    dem_raster_band: int
        DEM raster band (starting from 1)

    Returns
    -------
    dem_interp: isce3.geometry.DEMInterpolator
        DEM interpolator for given DEM raster and geo grid.
        )");
}
