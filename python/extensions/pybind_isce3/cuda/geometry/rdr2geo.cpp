#include "rdr2geo.h"

#include <pybind11/numpy.h>
#include <string>

#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/io/Raster.h>
#include <isce3/product/RadarGridParameters.h>

using isce3::cuda::geometry::Topo;

namespace py = pybind11;

void addbinding(py::class_<Topo> & pyRdr2Geo)
{
    pyRdr2Geo
            .def(py::init([](const isce3::product::RadarGridParameters&
                                          radar_grid,
                                  const isce3::core::Orbit& orbit,
                                  const isce3::core::Ellipsoid& ellipsoid,
                                  const isce3::core::LUT2d<double>& doppler,
                                  const double threshold, const int numiter,
                                  const int extraiter,
                                  const dataInterpMethod dem_interp_method,
                                  const int epsg_out, const bool compute_mask,
                                  const int lines_per_block) {
                auto rdr2geo_obj = Topo(radar_grid, orbit, ellipsoid, doppler);
                rdr2geo_obj.threshold(threshold);
                rdr2geo_obj.numiter(numiter);
                rdr2geo_obj.extraiter(extraiter);
                rdr2geo_obj.demMethod(dem_interp_method);
                rdr2geo_obj.epsgOut(epsg_out);
                rdr2geo_obj.computeMask(compute_mask);
                rdr2geo_obj.linesPerBlock(lines_per_block);
                return rdr2geo_obj;
            }),
                    py::arg("radar_grid"), py::arg("orbit"),
                    py::arg("ellipsoid"),
                    py::arg("doppler") = isce3::core::LUT2d<double>(),
                    py::arg("threshold") = 0.05, py::arg("numiter") = 25,
                    py::arg("extraiter") = 10,
                    py::arg("dem_interp_method") =
                            isce3::core::BIQUINTIC_METHOD,
                    py::arg("epsg_out") = 4326, py::arg("compute_mask") = true,
                    py::arg("lines_per_block") = 1000)
            .def("topo",
                    py::overload_cast<isce3::io::Raster&, const std::string&>(
                            &Topo::topo),
                    py::arg("dem_raster"), py::arg("outdir"))
            .def("topo",
                    py::overload_cast<isce3::io::Raster&,
                                      isce3::io::Raster*,
                                      isce3::io::Raster*,
                                      isce3::io::Raster*,
                                      isce3::io::Raster*,
                                      isce3::io::Raster*,
                                      isce3::io::Raster*,
                                      isce3::io::Raster*,
                                      isce3::io::Raster*,
                                      isce3::io::Raster*,
                                      isce3::io::Raster*,
                                      isce3::io::Raster*>
                            (&Topo::topo),
                    py::arg("dem_raster"),
                    py::arg("x_raster"),
                    py::arg("y_raster"),
                    py::arg("height_raster"),
                    py::arg("incidence_angle_raster"),
                    py::arg("heading_angle_raster"),
                    py::arg("local_incidence_angle_raster"),
                    py::arg("local_Psi_raster"),
                    py::arg("simulated_amplitude_raster"),
                    py::arg("layover_shadow_raster"),
                    py::arg("ground_to_sat_east_raster"),
                    py::arg("ground_to_sat_north_raster"),
                    R"(
        Run topo and output to user created topo rasters

        Parameters
        ----------
        dem_raster: isce3.io.Raster
            Input DEM raster
        x_raster: isce3.io.Raster
            Output raster for X coordinate in requested projection system (meters or degrees)
        y_raster: isce3.io.Raster
            Output raster for Y cooordinate in requested projection system (meters or degrees)
        height_raster: isce3.io.Raster
            Output raster for height above ellipsoid (meters)
        incidence_raster: isce3.io.Raster
            Output raster for incidence angle (degrees) computed from vertical at target
        heading_angle_raster: isce3.io.Raster
            Output raster for azimuth angle (degrees) computed anti-clockwise from EAST (Right hand rule)
        local_incidence_raster: isce3.io.Raster
            Output raster for local incidence angle (degrees) at target
        local_psi_raster: isce3.io.Raster
            Output raster for local projection angle (degrees) at target
        simulated_amplitude_raster: isce3.io.Raster
            Output raster for simulated amplitude image.
        layover_shadow_raster: isce3.io.Raster
            Output raster for layover/shadow mask.
        ground_to_sat_east_raster: isce3.io.Raster
            Output raster for east component of ground to satellite unit vector
        ground_to_sat_north_raster: isce3.io.Raster
            Output raster for north component of ground to satellite unit vector
                    )")
            .def_property_readonly("orbit", &Topo::orbit)
            .def_property_readonly("ellipsoid", &Topo::ellipsoid)
            .def_property_readonly("doppler", &Topo::doppler)
            .def_property_readonly("radar_grid", &Topo::radarGridParameters)
            .def_property("threshold",
                    py::overload_cast<>(&Topo::threshold, py::const_),
                    py::overload_cast<double>(&Topo::threshold))
            .def_property("numiter",
                    py::overload_cast<>(&Topo::numiter, py::const_),
                    py::overload_cast<int>(&Topo::numiter))
            .def_property("extraiter",
                    py::overload_cast<>(&Topo::extraiter, py::const_),
                    py::overload_cast<int>(&Topo::extraiter))
            .def_property("dem_interp_method",
                    py::overload_cast<>(&Topo::demMethod, py::const_),
                    py::overload_cast<dataInterpMethod>(&Topo::demMethod))
            .def_property("epsg_out",
                    py::overload_cast<>(&Topo::epsgOut, py::const_),
                    py::overload_cast<int>(&Topo::epsgOut))
            .def_property("compute_mask",
                    py::overload_cast<>(&Topo::computeMask, py::const_),
                    py::overload_cast<bool>(&Topo::computeMask))
            .def_property("lines_per_block",
                    py::overload_cast<>(&Topo::linesPerBlock, py::const_),
                    py::overload_cast<size_t>(&Topo::linesPerBlock))
            ;
}
