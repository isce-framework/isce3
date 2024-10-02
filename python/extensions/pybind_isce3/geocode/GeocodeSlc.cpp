#include "GeocodeSlc.h"

#include <isce3/core/Ellipsoid.h>
#include <isce3/core/EMatrix.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Poly2d.h>
#include <isce3/geocode/geocodeSlc.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/product/RadarGridParameters.h>
#include <isce3/product/SubSwaths.h>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

template<typename AzRgFunc = isce3::core::Poly2d>
void addbinding_geocodeslc(py::module & m)
{
    m.def("geocode_slc", py::overload_cast<isce3::io::Raster &,
            isce3::io::Raster &, isce3::io::Raster &,
            const isce3::product::RadarGridParameters &,
            const isce3::product::GeoGridParameters &,
            const isce3::core::Orbit &,
            const isce3::core::LUT2d<double> &,
            const isce3::core::LUT2d<double> &,
            const isce3::core::Ellipsoid &,
            const double &, const int &,
            const size_t &,
            const bool,
            const bool,
            const AzRgFunc &,
            const AzRgFunc &,
            const isce3::core::LUT2d<double> &,
            const isce3::core::LUT2d<double> &,
            const bool,
            const std::complex<float>,
            isce3::io::Raster*,
            isce3::io::Raster*>(&isce3::geocode::geocodeSlc<AzRgFunc>),
        py::arg("output_raster"),
        py::arg("input_raster"),
        py::arg("dem_raster"),
        py::arg("radargrid"),
        py::arg("geogrid"),
        py::arg("orbit"),
        py::arg("native_doppler"),
        py::arg("image_grid_doppler"),
        py::arg("ellipsoid"),
        py::arg("threshold_geo2rdr") = 1.0e-9,
        py::arg("numiter_geo2rdr") = 25,
        py::arg("lines_per_block") = 1000,
        py::arg("flatten") = true,
        py::arg("reramp") = true,
        py::arg("azimuth_carrier") = AzRgFunc(),
        py::arg("range_carrier") = AzRgFunc(),
        py::arg("az_time_correction") = isce3::core::LUT2d<double>(),
        py::arg("srange_correction") = isce3::core::LUT2d<double>(),
        py::arg("flatten_with_corrected_srange") = false,
        py::arg("invalid_value") =
            std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()),
        py::arg("carrier_phase_raster") = nullptr,
        py::arg("flatten_phase_raster") = nullptr,
        R"(
        Geocode a SLC raster

        Parameters
        ----------
        output_raster: Raster
            Output raster containing geocoded SLC
        input_raster: Raster
            Input raster of the SLC in radar coordinates
        dem_raster: Raster
            Raster of the DEM
        radargrid: RadarGridParameters
            Radar grid parameters of input SLC raster
        geogrid: GeoGridParameters
            Geo grid parameters of output raster
        orbit: isce3.core.Orbit
            Orbit object associated with radar grid
        native_doppler: LUT2d
            2D LUT doppler of the SLC image
        image_grid_doppler: LUT2d
            2d LUT doppler of the image grid
        ellipsoid: Ellipsoid
            Ellipsoid object
        threshold_geo2rdr: float
            Threshold for geo2rdr computations
        numiter_geo2rdr: int
            Maximum number of iterations for geo2rdr convergence
        lines_per_block: int
            Number of lines per block
        flatten: bool
            Flag to flatten the geocoded SLC
        reramp: bool
            Flag to reramp the geocoded SLC
        azimuth_carrier: [LUT2d, Poly2d]
            Azimuth carrier phase of the SLC data, in radians, as a function of azimuth and range
        range_carrier: [LUT2d, Poly2d]
            Range carrier phase of the SLC data, in radians, as a function of azimuth and range
        az_time_correction: LUT2d
             geo2rdr azimuth additive correction, in seconds, as a function of azimuth and range
        srange_correction: LUT2d
            geo2rdr slant range additive correction, in meters, as a function of azimuth and range
        flatten_with_corrected_srange: bool
            flag to indicate whether geo2rdr slant-range additive values should be used for phase flattening
        invalid_value: complex
            invalid pixel fill value
        carrier_phase_raster: Raster
            Optional output raster containing geocoded carrier phase
        flatten_phase_raster: Raster
            Optional output raster containing geocoded flattening phase
        )");
    m.def("geocode_slc", py::overload_cast<isce3::io::Raster &,
            isce3::io::Raster &, isce3::io::Raster &,
            const isce3::product::RadarGridParameters &,
            const isce3::product::RadarGridParameters &,
            const isce3::product::GeoGridParameters &,
            const isce3::core::Orbit &,
            const isce3::core::LUT2d<double> &,
            const isce3::core::LUT2d<double> &,
            const isce3::core::Ellipsoid &,
            const double &, const int &,
            const size_t &,
            const bool,
            const bool,
            const AzRgFunc &,
            const AzRgFunc &,
            const isce3::core::LUT2d<double> &,
            const isce3::core::LUT2d<double> &,
            const bool,
            const std::complex<float>,
            isce3::io::Raster*,
            isce3::io::Raster*>(&isce3::geocode::geocodeSlc<AzRgFunc>),
        py::arg("output_raster"),
        py::arg("input_raster"),
        py::arg("dem_raster"),
        py::arg("radargrid"),
        py::arg("sliced_radargrid"),
        py::arg("geogrid"),
        py::arg("orbit"),
        py::arg("native_doppler"),
        py::arg("image_grid_doppler"),
        py::arg("ellipsoid"),
        py::arg("threshold_geo2rdr") = 1.0e-9,
        py::arg("numiter_geo2rdr") = 25,
        py::arg("lines_per_block") = 1000,
        py::arg("flatten") = true,
        py::arg("reramp") = true,
        py::arg("azimuth_carrier") = AzRgFunc(),
        py::arg("range_carrier") = AzRgFunc(),
        py::arg("az_time_correction") = isce3::core::LUT2d<double>(),
        py::arg("srange_correction") = isce3::core::LUT2d<double>(),
        py::arg("flatten_with_corrected_srange") = false,
        py::arg("invalid_value") =
            std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()),
        py::arg("carrier_phase_raster") = nullptr,
        py::arg("flatten_phase_raster") = nullptr,
        R"(
        Geocode a subset of a SLC raster based a sliced radar grid

        Parameters
        ----------
        output_raster: Raster
            Output raster containing geocoded SLC
        input_raster: Raster
            Input raster of the SLC in radar coordinates
        dem_raster: Raster
            Raster of the DEM
        radargrid: RadarGridParameters
            Radar grid parameters of input SLC raster
        sliced_radargrid: RadarGridParameters
            Radar grid representing subset of radargrid
        geogrid: GeoGridParameters
            Geo grid parameters of output raster
        orbit: isce3.core.Orbit
            Orbit object associated with radar grid
        native_doppler: LUT2d
            2D LUT doppler of the SLC image
        image_grid_doppler: LUT2d
            2d LUT doppler of the image grid
        ellipsoid: Ellipsoid
            Ellipsoid object
        threshold_geo2rdr: float
            Threshold for geo2rdr computations
        numiter_geo2rdr: int
            Maximum number of iterations for geo2rdr convergence
        lines_per_block: int
            Number of lines per block
        flatten: bool
            Flag to flatten the geocoded SLC
        reramp: bool
            Flag to reramp the geocoded SLC
        azimuth_carrier: [LUT2d, Poly2d]
            Azimuth carrier phase of the SLC data, in radians, as a function of azimuth and range
        range_carrier: [LUT2d, Poly2d]
            Range carrier phase of the SLC data, in radians, as a function of azimuth and range
        az_time_correction: LUT2d
             geo2rdr azimuth additive correction, in seconds, as a function of azimuth and range
        srange_correction: LUT2d
            geo2rdr slant range additive correction, in meters, as a function of azimuth and range
        flatten_with_corrected_srange: bool
            flag to indicate whether geo2rdr slant-range additive values should be used for phase flattening
        invalid_value: complex
            invalid pixel fill value
        carrier_phase_raster: Raster
            Optional output raster containing geocoded carrier phase
        flatten_phase_raster: Raster
            Optional output raster containing geocoded flattening phase
        )");
    m.def("_geocode_slc", py::overload_cast<
            std::vector<isce3::geocode::EArray2dc64>&,
            isce3::geocode::EArray2df64,
            isce3::geocode::EArray2df64,
            const std::vector<isce3::geocode::EArray2dc64>&,
            isce3::io::Raster&,
            const isce3::product::RadarGridParameters&,
            const isce3::product::RadarGridParameters&,
            const isce3::product::GeoGridParameters&,
            const isce3::core::Orbit&,
            const isce3::core::LUT2d<double>&,
            const isce3::core::LUT2d<double>&,
            const isce3::core::Ellipsoid&,
            const double&, const int&,
            isce3::geocode::EArray2duc8,
            const size_t&, const size_t&,
            const bool,
            const bool,
            const AzRgFunc&,
            const AzRgFunc&,
            const isce3::core::LUT2d<double> &,
            const isce3::core::LUT2d<double> &,
            const bool,
            const std::complex<float>,
            const isce3::product::SubSwaths*>
            (&isce3::geocode::geocodeSlc<AzRgFunc>),
        py::arg("geo_data_blocks"),
        py::arg("carrier_phase_block"),
        py::arg("flatten_phase_block"),
        py::arg("rdr_data_blocks"),
        py::arg("dem_raster"),
        py::arg("radargrid"),
        py::arg("sliced_radar_grid"),
        py::arg("geogrid"),
        py::arg("orbit"),
        py::arg("native_doppler"),
        py::arg("image_grid_doppler"),
        py::arg("ellipsoid"),
        py::arg("threshold_geo2rdr"), py::arg("numiter_geo2rdr"),
        py::arg("mask_block") = nullptr,
        py::arg("azimuth_first_line") = 0, py::arg("range_first_pixel") = 0,
        py::arg("flatten") = true,
        py::arg("reramp") = true,
        py::arg("az_carrier") = AzRgFunc(),
        py::arg("rg_carrier") = AzRgFunc(),
        py::arg("az_time_correction") = isce3::core::LUT2d<double>(),
        py::arg("srange_correction") = isce3::core::LUT2d<double>(),
        py::arg("flatten_with_corrected_srange") = false,
        py::arg("invalid_value") =
            std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()),
        py::arg("subswaths") = nullptr,
        R"(
        Geocode a subset of pixels for multiple radar SLC arrays to a given
        geogrid. All radar SLC arrays share a common radar grid. All output
        geocoded arrays share a common geogrid. Subset of pixels defined by a
        sliced radar grid - a radar grid contained within the common radar
        grid.

        Parameters
        ----------
        geo_data_blocks: list of numpy.ndarray
            List of output arrays containing geocoded SLC
        mask_block: numpy.ndarray
            Output array containing masking values of geocoded SLC
        carrier_phase_block: numpy.ndarray
            Output array containing geocoded carrier phase
        flatten_phase_block: numpy.ndarray
            Output array containing geocoded flattening phase
        rdr_data_blocks: list of numpy.ndarray
            List of input arrays of the SLC in radar coordinates
        dem_raster: Raster
            Raster of the DEM
        radargrid: RadarGridParameters
            Radar grid parameters of input SLC raster
        sliced_radargrid: RadarGridParameters
            Radar grid representing subset of radargrid
        geogrid: GeoGridParameters
            Geo grid parameters of output raster
        orbit: isce3.core.Orbit
            Orbit object associated with radar grid
        native_doppler: LUT2d
            2D LUT doppler of the SLC image
        image_grid_doppler: LUT2d
            2d LUT doppler of the image grid
        ellipsoid: Ellipsoid
            Ellipsoid object
        threshold_geo2rdr: float
            Threshold for geo2rdr computations
        numiter_geo2rdr: int
            Maximum number of iterations for geo2rdr convergence
        azimuth_first_line: int
            FIrst line of radar data block with respect to larger radar data raster, else 0
        range_first_pixel: int
            FIrst pixel of radar data block with respect to larger radar data raster, else 0
        flatten: bool
            Flag to flatten the geocoded SLC
        reramp: bool
            Flag to reramp the geocoded SLC
        azimuth_carrier: [LUT2d, Poly2d]
            Azimuth carrier phase of the SLC data, in radians, as a function of azimuth and range
        range_carrier: [LUT2d, Poly2d]
            Range carrier phase of the SLC data, in radians, as a function of azimuth and range
        az_time_correction: LUT2d
             geo2rdr azimuth additive correction, in seconds, as a function of azimuth and range
        srange_correction: LUT2d
            geo2rdr slant range additive correction, in meters, as a function of azimuth and range
        flatten_with_corrected_srange: bool
            flag to indicate whether geo2rdr slant-range additive values should be used for phase flattening
        invalid_value: complex
            invalid pixel fill value
        subswaths: isce3.product.SubSwaths, optional
            SubSwaths from RSLC to be used for masking geocoded output. If None,
            no subswath masking is performed. Defaults to None.
        )");
}

template void addbinding_geocodeslc<isce3::core::LUT2d<double>>(py::module & m);
template void addbinding_geocodeslc<isce3::core::Poly2d>(py::module & m);
