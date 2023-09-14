#include "getRadarGrid.h"

#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/geometry/detail/Geo2Rdr.h>

namespace py = pybind11;

void addbinding_get_radar_grid(pybind11::module& m)
{

    const isce3::geometry::detail::Geo2RdrParams geo2rdr_defaults;

    m.def("get_radar_grid", &isce3::geogrid::getRadarGrid,
          py::arg("lookside"), py::arg("wavelength"), py::arg("dem_raster"),
          py::arg("geogrid"), py::arg("orbit"), py::arg("native_doppler"), 
          py::arg("grid_doppler"),
          py::arg("dem_interp_method") = isce3::core::BIQUINTIC_METHOD,
          py::arg("geo2rdr_params") = geo2rdr_defaults,
          py::arg("interpolated_dem_raster") = nullptr,
          py::arg("slant_range_raster") = nullptr,
          py::arg("azimuth_time_raster") = nullptr,
          py::arg("incidence_angle_raster") = nullptr,
          py::arg("los_unit_vector_x_raster") = nullptr,
          py::arg("los_unit_vector_y_raster") = nullptr,
          py::arg("along_track_unit_vector_x_raster") = nullptr,
          py::arg("along_track_unit_vector_y_raster") = nullptr,
          py::arg("elevation_angle_raster") = nullptr,
          py::arg("ground_track_velocity_raster") = nullptr,
          py::arg("local_incidence_angle_raster") = nullptr,
          py::arg("projection_angle_raster") = nullptr,
          py::arg("simulated_radar_brightness_raster") = nullptr,
          R"(Get radar grid from L2 products

             Each output layer is saved onto the first band of its
             associated raster file.
             
             The target-to-sensor line-of-sight (LOS) and along-track unit
             vectors are referenced to ENU coordinates computed wrt targets.

             Parameters
             ----------
             lookside : isce3.core.LookSide
                 Look side
             wavelength : double
                 Wavelength
             dem_raster : isce3.io.Raster
                 DEM raster
             geogrid : isce3.product.GeoGridParameters
                 Output layers geogrid
             orbit : isce3.core.Orbit
                 Orbit
             native_doppler : isce3.core.LUT2d
                 Native image Doppler
             grid_doppler : isce3.core.LUT2d
                 Grid Doppler
             dem_interp_method :  isce3::core::dataInterpMethod
                 DEM interpolation method (default: biquintic)
             geo2rdr_params : double, optional
                 Geo2rdr parameters
             interpolated_dem_raster : isce3.io.Raster, optional
                 Interpolated DEM raster
             slant_range_raster : isce3.io.Raster, optional
                 Slant-range (in meters) cube raster
             azimuth_time_raster : isce3.io.Raster, optional
                 Azimuth time (in seconds relative to orbit epoch) cube raster
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
             local_incidence_angle_raster : isce3.io.Raster, optional
                 Local-incidence angle raster
             projection_angle_raster : isce3.io.Raster, optional
                 Projection angle raster
             simulated_radar_brightness_raster : isce3.io.Raster, optional
                 Simulated radar brightness raster
)");

}
