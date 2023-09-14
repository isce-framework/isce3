#include "getGeolocationGrid.h"

#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/detail/Rdr2Geo.h>

namespace py = pybind11;

void addbinding_get_geolocation_grid(pybind11::module& m)
{
    const isce3::geometry::detail::Geo2RdrParams geo2rdr_defaults;
    const isce3::geometry::detail::Rdr2GeoParams rdr2geo_defaults;

    m.def("get_geolocation_grid", &isce3::geometry::getGeolocationGrid,
          py::arg("dem_raster"),
          py::arg("radar_grid"), py::arg("orbit"), py::arg("native_doppler"), 
          py::arg("grid_doppler"), py::arg("epsg"),
          py::arg("dem_interp_method") = isce3::core::BIQUINTIC_METHOD,
          py::arg("rdr2geo_params") = rdr2geo_defaults,
          py::arg("geo2rdr_params") = geo2rdr_defaults,
          py::arg("interpolated_dem_raster") = nullptr,
          py::arg("coordiante_x_raster") = nullptr,
          py::arg("coordinate_y_raster") = nullptr,
          py::arg("incidence_angle_raster") = nullptr,
          py::arg("lo_unit_vector_x_raster") = nullptr,
          py::arg("los_unit_vector_y_raster") = nullptr,
          py::arg("along_track_unit_vector_x_raster") = nullptr,
          py::arg("along_track_unit_vector_y_raster") = nullptr,
          py::arg("elevation_angle_raster") = nullptr,
          py::arg("ground_track_velocity_raster") = nullptr,
          R"(Get geolocation grid from L1 products

            The target-to-sensor line-of-sight (LOS) and along-track unit vectors are
            referenced to ENU coordinates computed wrt targets.

        Parameters
        ----------
              dem_raster : isce3.io.Raster
                  DEM raster
              radar_grid : isce3.product.RadarGridParameters
                  Radar Grid
              orbit : isce3.core.Orbit
                  Orbit
              native_doppler : isce3.core.LUT2d
                  Native image Doppler
              grid_doppler : isce3.core.LUT2d
                  Grid Doppler
              epsg : int
                  Output geolocation EPSG
              dem_interp_method :  isce3::core::dataInterpMethod
                  DEM interpolation method
              rdr2geo_params : double, optional
                  Rdr2geo parameters
              geo2rdr_params : double, optional
                  Geo2rdr parameters
              interpolated_dem_raster : isce3.io.Raster, optional
                  Interpolated DEM raster
              coordiante_x_raster : isce3.io.Raster, optional
                  Coordinate-X raster
              coordinate_y_raster : isce3.io.Raster, optional
                  Coordiante-Y raster
              incidence_angle_raster : isce3.io.Raster, optional
                  Incidence angle (in degrees wrt ellipsoid normal at target) 
                  cube raster
              los_unit_vector_x_raster : isce3.io.Raster, optional
                  LOS (target-to-sensor) unit vector X cube raster
              los_unit_vector_y_raster : isce3.io.Raster, optional
                  LOS (target-to-sensor) unit vector Y cube raster
              along_track_unit_vector_x_raster : isce3.io.Raster, optional
                  Along-track unit vector X raster
              along_track_unit_vector_y_raster : isce3.io.Raster, optional
                  Along-track unit vector Y raster
              elevation_angle_raster : isce3.io.Raster, optional
                  Elevation angle (in degrees wrt geodedic nadir) cube raster
              ground_track_velocity_raster : isce3.io.Raster, optional
                  Ground-track velocity raster
)");

}

