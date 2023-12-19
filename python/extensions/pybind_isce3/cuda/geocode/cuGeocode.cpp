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
    const isce3::geometry::detail::Geo2RdrParams g2r_defaults;

    pyGeocode
    .def(py::init<const isce3::product::GeoGridParameters&,
                 const isce3::container::RadarGeometry&,
                 const size_t>(),
            py::arg("geogrid_params"),
            py::arg("radar_geometry"),
            py::arg("lines_per_block") = 1000,
            R"(
            Create CUDA geocode object.

            Parameters
            ----------
            geogrid_params: GeoGridParameters
                Geogrid defining output product
            radar_geometry: RadarGeometry
                Radar grid describing input rasters
            lines_per_block: int
                Number of lines per block to be processed. Defualt 1000
            )")
    .def("geocode_rasters", &Geocode::geocodeRasters,
            py::arg("output_rasters"),
            py::arg("input_rasters"),
            py::arg("data_interp_methods"),
            py::arg("raster_datatypes"),
            py::arg("invalid_values"),
            py::arg("dem_raster"),
            py::arg("native_doppler") = isce3::core::LUT2d<double>(),
            py::arg("az_time_correction") = isce3::core::LUT2d<double>(),
            py::arg("srange_correction") = isce3::core::LUT2d<double>(),
            py::arg("subswaths") = nullptr,
            py::arg("dem_interp_method") =
                    isce3::core::BIQUINTIC_METHOD,
            py::arg("threshold") = g2r_defaults.threshold,
            py::arg("maxiter") = g2r_defaults.maxiter,
            py::arg("delta_range") = g2r_defaults.delta_range,
            R"(
            Geocode rasters with a shared geogrid with block processing handled internally.

            Parameters
            ----------
            output_rasters: list(isce3.io.Raster)
                List of geocoded rasters.
            input_rasters: list(isce3.io.Raster)
                List of rasters to be geocoded.
            data_interp_methods: list[isce3.io.gdal.GDALDataType]
                Interpolation methods used to interpolate each raster
            raster_datatypes: list[int]
                Datatype of each raster to be geocoded
            invalid_values: list[np.float64]
                Invalid values for each geocoded raster
            dem_raster: isce3.io.Raster
                DEM used to calculate radar grid indices
            native_doppler: isce3.core.LUT2d
                Doppler centroid of data associated with radar grid, in Hz, as
                fuction of azimuth and range
            az_time_correction: isce3.core.LUT2d
                geo2rdr azimuth additive correction, in seconds, as a function
                of azimuth and range
            srange_correction: isce3.core.LUT2d
                geo2rdr slant range additive correction, in seconds, as a
                function of azimuth and range
            dem_interp_method: isce3.core.DataInterpMethod
                Interpolation method used by DEM interpolator. Default
                BIQUINTIC_METHOD
            threshold: double
                Convergence threshold for geo2rdr. Defualt 1e-8
            maxiter: int
                Maximum iterations for geo2rdr. Default 50
            delta_range: double
                Step size for numerical gradient for geo2rdr. Default 10
        )")
    ;
}
