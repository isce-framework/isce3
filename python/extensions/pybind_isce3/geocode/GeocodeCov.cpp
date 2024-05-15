#include "GeocodeCov.h"

#include <limits>

#include <optional>
#include <pybind11/stl.h>

#include <isce3/core/Constants.h>
#include <isce3/core/blockProcessing.h>
#include <isce3/geometry/RTC.h>
#include <isce3/io/Raster.h>

namespace py = pybind11;

using isce3::core::parseDataInterpMethod;
using isce3::geocode::Geocode;
using isce3::core::GeocodeMemoryMode;
using isce3::geocode::geocodeOutputMode;
using isce3::geometry::rtcAlgorithm;
using isce3::geometry::rtcAreaBetaMode;
using isce3::geometry::rtcInputTerrainRadiometry;
using isce3::geometry::rtcOutputTerrainRadiometry;
using isce3::io::Raster;
using isce3::product::RadarGridParameters;

template<typename T>
void addbinding(py::class_<Geocode<T>>& pyGeocode)
{
    pyGeocode.def(py::init<>())
            .def_property("orbit", nullptr, &Geocode<T>::orbit)
            .def_property("doppler", nullptr, &Geocode<T>::doppler)
            .def_property("native_doppler", nullptr, &Geocode<T>::nativeDoppler)
            .def_property("ellipsoid", nullptr, &Geocode<T>::ellipsoid)
            .def_property("threshold_geo2rdr", nullptr,
                          &Geocode<T>::thresholdGeo2rdr)
            .def_property("numiter_geo2rdr", nullptr,
                          &Geocode<T>::numiterGeo2rdr)
            .def_property("radar_block_margin", nullptr,
                    &Geocode<T>::radarBlockMargin)
            .def_property("data_interpolator",
                    py::overload_cast<>(
                            &Geocode<T>::dataInterpolator, py::const_),
                    [](Geocode<T>& self, std::string& method) {
                        // get interp method
                        auto data_interpolator = parseDataInterpMethod(method);

                        // set interp method
                        self.dataInterpolator(data_interpolator);
                    })
            .def_property_readonly(
                    "geogrid_start_x", &Geocode<T>::geoGridStartX)
            .def_property_readonly(
                    "geogrid_start_y", &Geocode<T>::geoGridStartY)
            .def_property_readonly(
                    "geogrid_spacing_x", &Geocode<T>::geoGridSpacingX)
            .def_property_readonly(
                    "geogrid_spacing_y", &Geocode<T>::geoGridSpacingY)
            .def_property_readonly("geogrid_width", &Geocode<T>::geoGridWidth)
            .def_property_readonly("geogrid_length", &Geocode<T>::geoGridLength)
            .def("update_geogrid", &Geocode<T>::updateGeoGrid,
                    py::arg("radar_grid"), py::arg("dem_raster"))
            .def("geogrid", &Geocode<T>::geoGrid, py::arg("x_start"),
                    py::arg("y_start"), py::arg("x_spacing"),
                    py::arg("y_spacing"), py::arg("width"), py::arg("length"),
                    py::arg("epsg"))
            .def("geocode", &Geocode<T>::geocode, py::arg("radar_grid"),
                    py::arg("input_raster"), py::arg("output_raster"),
                    py::arg("dem_raster"),
                    py::arg("output_mode") = geocodeOutputMode::AREA_PROJECTION,
                    py::arg("flag_az_baseband_doppler") = false,
                    py::arg("flatten") = false,
                    py::arg("geogrid_upsampling") = 1,
                    py::arg("flag_upsample_radar_grid") = false,
                    py::arg("flag_apply_rtc") = false,
                    py::arg("input_terrain_radiometry") =
                            rtcInputTerrainRadiometry::BETA_NAUGHT,
                    py::arg("output_terrain_radiometry") =
                            rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
                    py::arg("exponent") = 0,
                    py::arg("rtc_min_value_db") =
                            std::numeric_limits<float>::quiet_NaN(),
                    py::arg("rtc_upsampling") =
                            std::numeric_limits<double>::quiet_NaN(),
                    py::arg("rtc_algorithm") =
                            rtcAlgorithm::RTC_AREA_PROJECTION,
                    py::arg("rtc_area_beta_mode") =
                            rtcAreaBetaMode::AUTO,
                    py::arg("abs_cal_factor") = 1,
                    py::arg("clip_min") =
                            std::numeric_limits<float>::quiet_NaN(),
                    py::arg("clip_max") =
                            std::numeric_limits<float>::quiet_NaN(),
                    py::arg("min_nlooks") =
                            std::numeric_limits<float>::quiet_NaN(),
                    py::arg("radargrid_nlooks") = 1,
                    py::arg("out_off_diag_terms") = nullptr,
                    py::arg("out_geo_rdr") = nullptr,
                    py::arg("out_geo_dem") = nullptr,
                    py::arg("out_geo_nlooks") = nullptr,
                    py::arg("out_geo_rtc") = nullptr,
                    py::arg("out_geo_rtc_gamma0_to_sigma0") = nullptr,
                    py::arg("phase_screen") = nullptr,
                    py::arg("az_time_correction") = isce3::core::LUT2d<double>(),
                    py::arg("slant_range_correction") = isce3::core::LUT2d<double>(),
                    py::arg("input_rtc") = nullptr,
                    py::arg("output_rtc") = nullptr,
                    py::arg("input_layover_shadow_mask_raster") = nullptr,
                    py::arg("sub_swaths") = nullptr,
                    py::arg("apply_valid_samples_sub_swath_masking") = std::nullopt,
                    py::arg("out_mask") = nullptr,
                    py::arg("memory_mode") = GeocodeMemoryMode::Auto,
                    py::arg("min_block_size") =
                            isce3::core::DEFAULT_MIN_BLOCK_SIZE,
                    py::arg("max_block_size") =
                            isce3::core::DEFAULT_MAX_BLOCK_SIZE,
                    py::arg("dem_interp_method") =
                            isce3::core::BIQUINTIC_METHOD,
                    R"(
                    Geocode data from slant-range to map coordinates

                    Parameters
                    ----------
                    radar_grid: isce3.product.RadarGridParameters
                        Radar grid
                    input_raster: isce3.io.Raster
                        Input raster
                    output_raster: isce3.io.Raster
                        Output raster
                    dem_raster: isce3.io.Raster
                        Input DEM raster
                    output_mode: isce3.geocode.GeocodeOutputMode
                        Geocode method
                    flag_az_baseband_doppler: bool, optional
                        Shift SLC azimuth spectrum to baseband (using Doppler
                        centroid) before interpolation
                    flatten: bool, optional
                        Flatten the geocoded SLC
                    geogrid_upsampling: int, optional
                        Geogrid upsampling
                    flag_upsample_radar_grid: bool, optional
                        Double the radar grid sampling rate
                    flag_apply_rtc: bool, optional
                        Apply radiometric terrain correction (RTC)
                    input_terrain_radiometry: isce3.geometry.RtcInputTerrainRadiometry, optional
                        Input terrain radiometry
                    output_terrain_radiometry: isce3.geometry.RtcOutputTerrainRadiometry, optional
                        Output terrain radiometry
                    exponent: int, optional
                        Exponent to be applied to the input data. The value 0
                        indicates that the exponent is based on the data type
                        of the input raster (1 for real and 2 for complex rasters).
                    rtc_min_value_db: float, optional
                        Minimum value for the RTC area factor. Radar data with
                        RTC area factor below this limit will be set to NaN.
                    rtc_geogrid_upsampling: int, optional
                        Geogrid upsampling to compute the radiometric terrain
                        correction RTC.
                    rtc_algorithm: isce3.geometry.RtcAlgorithm, optional
                        RTC algorithm
                    rtc_factor_area_mode : isce3.geometry.RtcAreaBetaMode, optional
                        RTC area beta mode
                    abs_cal_factor: float, optional
                        Absolute calibration factor.
                    clip_min: float, optional
                        Clip (limit) minimum output values
                    clip_max: float, optional
                        Clip (limit) maximum output values
                    min_nlooks: float, optional
                        Minimum number of looks. Geogrid data below this limit
                        will be set to NaN
                    radar_grid_nlooks: float, optional
                        Radar grid number of looks. This parameters determines
                        the multilooking factor used to compute out_geo_nlooks.
                    out_off_diag_terms: isce3.io.Raster, optional
                        Output raster containing the off-diagonal terms of the
                        covariance matrix.
                    out_geo_rdr: isce3.io.Raster, optional
                        Raster to which the radar-grid positions (range and
                        azimuth) of the geogrid pixels vertices will be saved.
                    out_geo_dem: isce3.io.Raster, optional
                        Raster to which the interpolated DEM will be saved.
                    out_nlooks: isce3.io.Raster, optional
                        Raster to which the number of radar-grid looks
                        associated with the geogrid will be saved.
                    out_geo_rtc: isce3.io.Raster, optional
                        Output RTC area factor (in geo-coordinates).
                    out_geo_rtc_gamma0_to_sigma0: isce3.io.Raster, optional
                        Output RTC area factor gamma0 to sigma0 array
                        (in geo-coordinates).
                    phase_screen_raster: isce3.io.Raster, optional
                        Phase screen to be removed before geocoding
                    az_time_correction: LUT2d
                        geo2rdr azimuth additive correction, in seconds,
                        as a function of azimuth and range
                    slant_range_correction: LUT2d
                        geo2rdr slant range additive correction, in meters,
                        as a function of azimuth and range
                    in_rtc: isce3.io.Raster, optional
                        Input RTC area factor (in slant-range).
                    output_rtc: isce3.io.Raster, optional
                        Output RTC area factor (in slant-range).
                    input_layover_shadow_mask_raster: isce3.io.Raster, optional
                        Input layover/shadow mask raster (in radar geometry).
                        Samples identified as SHADOW or LAYOVER_AND_SHADOW are
                        considered invalid.
                    sub_swaths: isce3.product.SubSwaths, optional
                        Sub-swaths metadata
                    apply_valid_samples_sub_swath_masking: bool, optional
                        Flag indicating whether the valid-samples sub-swath
                        masking should be applied during geocoding.
                        If not given, then sub-swath masking will be applied
                        if the sub_swaths parameter is provided.
                    out_mask: isce3.io.Raster, optional
                        Output valid-pixels sub-swath mask
                    geocode_memory_mode: isce3.core.GeocodeMemoryMode
                        Select memory mode
                    min_block_size: int, optional
                        Minimum block size (per thread)
                    max_block_size: int, optional
                        Maximum block size (per thread)
                    dem_interp_method: isce3.core.DataInterpMethod, optional
                        DEM interpolation method
                    )");
}

void addbinding(pybind11::enum_<geocodeOutputMode>& pyGeocodeOutputMode)
{
    pyGeocodeOutputMode.value("INTERP", geocodeOutputMode::INTERP)
            .value("AREA_PROJECTION", geocodeOutputMode::AREA_PROJECTION);
};

template void addbinding(py::class_<Geocode<float>>&);
template void addbinding(py::class_<Geocode<double>>&);
template void addbinding(py::class_<Geocode<std::complex<float>>>&);
template void addbinding(py::class_<Geocode<std::complex<double>>>&);
// end of file
