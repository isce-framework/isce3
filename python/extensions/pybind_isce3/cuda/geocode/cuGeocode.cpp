#include "cuGeocode.h"

#include <gdal_priv.h>
#include <pybind11/stl.h>

#include <isce3/container/RadarGeometry.h>
#include <isce3/core/Constants.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>

namespace py = pybind11;

using isce3::cuda::geocode::Geocode;

void addbinding(pybind11::class_<Geocode>& pyGeocode)
{
    const isce3::geometry::detail::Geo2RdrParams defaults;
    pyGeocode
            .def(py::init<const isce3::product::GeoGridParameters&,
                         const isce3::container::RadarGeometry&,
                         const isce3::io::Raster&, const double, const size_t,
                         const isce3::core::dataInterpMethod,
                         const isce3::core::dataInterpMethod, const double,
                         const int, const double, const float>(),
                    py::arg("geogrid_params"), py::arg("radar_geometry"),
                    py::arg("dem_raster"), py::arg("dem_margin"),
                    py::arg("lines_per_block") = 1000,
                    py::arg("data_interp_method") =
                            isce3::core::BILINEAR_METHOD,
                    py::arg("dem_interp_method") =
                            isce3::core::BIQUINTIC_METHOD,
                    py::arg("threshold") = defaults.threshold,
                    py::arg("maxiter") = defaults.maxiter,
                    py::arg("delta_range") = defaults.delta_range,
                    py::arg("invalid_value") = 0.0,
                    R"(
            Create CUDA geocode object.

            Parameters
            ----------
            geogrid_params: GeoGridParameters
                Geogrid defining output product
            radar_geometry: RadarGeometry
                Radar grid describing input rasters
            dem_raster: Raster
                DEM used to calculate radar grid indices
            dem_margin: double
                Extra margin applied to bounding box used to load DEM. Units
                need to match geogrid_params EPSG units.
            lines_per_block: int
                Number of lines to be processed
                Defualt 1000
            data_interp_method: enum
                Interpolation method used by data interpolator
            dem_interp_method: enum
                Interpolation method used by DEM interpolator
            threshold: double
                Convergence threshold for geo2rdr
            maxiter: int
                Maximum iterations for geo2rdr
            delta_range: double
                Step size for numerical gradient for geo2rdr
            invalid_value: float
                Value assigned to invalid geogrid pixels
            )")
            .def("set_block_radar_coord_grid", &Geocode::setBlockRdrCoordGrid,
                    py::arg("block_number"),
                    R"(
            Calculate set radar grid coordinates of geocode grid for a given block
            number.

            Parameters
            ----------
            block_number: int
                Index of block of raster where radar grid coordinates are calculated
                and set.
            )")
    .def("geocode_raster_block", [](Geocode & self,
                                    isce3::io::Raster & output_raster,
                                    isce3::io::Raster & input_raster) {
                const int dtype =  input_raster.dtype();
                switch (dtype) {
                    case GDT_Float32:   {
                        self.geocodeRasterBlock<float>(
                                output_raster, input_raster);
                        break; }
                    case GDT_CFloat32:  {
                        self.geocodeRasterBlock<thrust::complex<float>>(
                                output_raster, input_raster);
                        break;}
                    case GDT_Float64:   {
                        self.geocodeRasterBlock<double>(
                                output_raster, input_raster);
                        break; }
                    case GDT_CFloat64:  {
                        self.geocodeRasterBlock<thrust::complex<double>>(
                                output_raster, input_raster);
                        break;}
                    case GDT_Byte:  {
                        self.geocodeRasterBlock<unsigned char>(
                                output_raster, input_raster);
                        break;}
                    case GDT_UInt32:  {
                        self.geocodeRasterBlock<unsigned int>(
                                output_raster, input_raster);
                        break;}
                    default: {
                        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                "unsupported datatype");
                             }
                }
            },
            py::arg("output_raster"),
            py::arg("input_raster"),
            R"(
            Geocode raster according to block specified in make_radar_grid_coordinates.

            Parameters
            ----------
            output_raster: io::Raster
                Geocoded raster
            input_raster: io::Raster
                Raster to be geocoded
        )")
    .def("geocode_rasters", &Geocode::geocodeRasters,
            py::arg("output_rasters"),
            py::arg("input_rasters"),
            R"(
            Geocode rasters with a shared geogrid with block processing handled internally.

            Parameters
            ----------
            output_rasters: list(io::Raster)
                List of geocoded rasters.
            input_rasters: list(io::Raster)
                List of rasters to be geocoded.
        )")
    .def_property_readonly("n_blocks", &Geocode::numBlocks)
    .def_property_readonly("lines_per_block", &Geocode::linesPerBlock);
    ;
}
