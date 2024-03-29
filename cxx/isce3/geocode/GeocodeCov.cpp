#include "GeocodeCov.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cpl_virtualmem.h>
#include <limits>

#include <isce3/core/Basis.h>
#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Projections.h>
#include <isce3/core/TypeTraits.h>
#include <isce3/core/Constants.h>
#include <isce3/core/Utilities.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/loadDem.h>
#include <isce3/geometry/RTC.h>
#include <isce3/geometry/boundingbox.h>
#include <isce3/geometry/geometry.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/signal/Looks.h>
#include <isce3/signal/signalUtils.h>

#include "GeocodeHelpers.h"

using isce3::core::OrbitInterpBorderMode;
using isce3::core::Vec3;
using isce3::core::GeocodeMemoryMode;

namespace isce3 { namespace geocode {

template<class T>
void Geocode<T>::updateGeoGrid(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& dem_raster)
{

    pyre::journal::info_t info("isce.geocode.GeocodeCov.updateGeoGrid");

    if (_epsgOut == 0)
        _epsgOut = dem_raster.getEPSG();

    if (std::isnan(_geoGridSpacingX))
        _geoGridSpacingX = dem_raster.dx();

    if (std::isnan(_geoGridSpacingY))
        _geoGridSpacingY = dem_raster.dy();

    if (std::isnan(_geoGridStartX) || std::isnan(_geoGridStartY) ||
        _geoGridLength <= 0 || _geoGridWidth <= 0) {
        std::unique_ptr<isce3::core::ProjectionBase> proj(
                isce3::core::createProj(_epsgOut));
        isce3::geometry::BoundingBox bbox =
                isce3::geometry::getGeoBoundingBoxHeightSearch(
                        radar_grid, _orbit, proj.get(), _doppler);
        _geoGridStartX = bbox.MinX;
        if (_geoGridSpacingY < 0)
            _geoGridStartY = bbox.MaxY;
        else
            _geoGridStartY = bbox.MinY;

        _geoGridWidth = (bbox.MaxX - bbox.MinX) / _geoGridSpacingX;
        _geoGridLength = std::abs((bbox.MaxY - bbox.MinY) / _geoGridSpacingY);
    }
}

template<class T>
void Geocode<T>::geoGrid(double geoGridStartX, double geoGridStartY,
                         double geoGridSpacingX, double geoGridSpacingY,
                         int width, int length, int epsgcode)
{

    // the starting coordinate of the output geocoded grid in X direction.
    _geoGridStartX = geoGridStartX;

    // the starting coordinate of the output geocoded grid in Y direction.
    _geoGridStartY = geoGridStartY;

    // spacing of the output geocoded grid in X
    _geoGridSpacingX = geoGridSpacingX;

    // spacing of the output geocoded grid in Y
    _geoGridSpacingY = geoGridSpacingY;

    // number of lines (rows) in the geocoded grid (Y direction)
    _geoGridLength = length;

    // number of columns in the geocoded grid (Y direction)
    _geoGridWidth = width;

    // Save the EPSG code
    _epsgOut = epsgcode;
}


static void _validateInputLayoverShadowMaskRaster(
        isce3::io::Raster* input_layover_shadow_mask_raster,
        const isce3::product::RadarGridParameters& radar_grid){

    pyre::journal::error_t error("isce3.geocode.GeocodeCov");

    if (input_layover_shadow_mask_raster->width() != radar_grid.width()) {
        std::string err_str {
            "ERROR the widths of the input layover/shadow mask (" +
            std::to_string(input_layover_shadow_mask_raster->width()) +
            ") and radar geometry (" +
            std::to_string(radar_grid.width()) +
            ") do not match"};
        error << err_str << pyre::journal::endl;
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), err_str);
    }

    if (input_layover_shadow_mask_raster->length() != radar_grid.length()) {
        std::string err_str {
            "ERROR the lengths of the input layover/shadow mask (" +
            std::to_string(input_layover_shadow_mask_raster->length()) +
            ") and radar geometry (" +
            std::to_string(radar_grid.length()) +
            ") do not match"};
        error << err_str << pyre::journal::endl;
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), err_str);
    }
}

template<class T>
void Geocode<T>::geocode(const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
        isce3::io::Raster& dem_raster, geocodeOutputMode output_mode,
        bool flag_az_baseband_doppler, bool flatten, double geogrid_upsampling,
        bool flag_upsample_radar_grid, bool flag_apply_rtc,
        isce3::geometry::rtcInputTerrainRadiometry input_terrain_radiometry,
        isce3::geometry::rtcOutputTerrainRadiometry output_terrain_radiometry,
        int exponent, float rtc_min_value_db, double rtc_geogrid_upsampling,
        isce3::geometry::rtcAlgorithm rtc_algorithm,
        isce3::geometry::rtcAreaBetaMode rtc_area_beta_mode,
        double abs_cal_factor, float clip_min, float clip_max,
        float min_nlooks, float radar_grid_nlooks,
        isce3::io::Raster* out_off_diag_terms,
        isce3::io::Raster* out_geo_rdr, isce3::io::Raster* out_geo_dem,
        isce3::io::Raster* out_geo_nlooks, isce3::io::Raster* out_geo_rtc,
        isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0,
        isce3::io::Raster* phase_screen_raster,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        isce3::io::Raster* input_rtc, isce3::io::Raster* output_rtc,
        isce3::io::Raster* input_layover_shadow_mask_raster,
        isce3::product::SubSwaths* sub_swaths,
        isce3::io::Raster* out_mask,
        GeocodeMemoryMode geocode_memory_mode,
        const long long min_block_size, const long long max_block_size,
        isce3::core::dataInterpMethod dem_interp_method)
{
    bool flag_complex_to_real = isce3::signal::verifyComplexToRealCasting(
            input_raster, output_raster, exponent);

    bool flag_run_geocode_interp = output_mode == geocodeOutputMode::INTERP;
    if (flag_run_geocode_interp && !flag_complex_to_real)
        geocodeInterp<T>(radar_grid, input_raster, output_raster, dem_raster,
                flag_apply_rtc, flag_az_baseband_doppler, flatten,
                input_terrain_radiometry, output_terrain_radiometry,
                rtc_min_value_db, rtc_geogrid_upsampling, rtc_algorithm,
                rtc_area_beta_mode,
                abs_cal_factor, clip_min, clip_max, out_geo_rdr, out_geo_dem,
                out_geo_rtc, out_geo_rtc_gamma0_to_sigma0,
                phase_screen_raster, az_time_correction,
                slant_range_correction, input_rtc, output_rtc,
                input_layover_shadow_mask_raster, sub_swaths,
                out_mask,
                geocode_memory_mode, min_block_size, max_block_size,
                dem_interp_method);
    else if (flag_run_geocode_interp &&
             (std::is_same<T, double>::value ||
                     std::is_same<T, std::complex<double>>::value))
        geocodeInterp<double>(radar_grid, input_raster, output_raster,
                dem_raster, flag_apply_rtc, flag_az_baseband_doppler, flatten,
                input_terrain_radiometry, output_terrain_radiometry,
                rtc_min_value_db, rtc_geogrid_upsampling, rtc_algorithm,
                rtc_area_beta_mode,
                abs_cal_factor, clip_min, clip_max, out_geo_rdr, out_geo_dem,
                out_geo_rtc, out_geo_rtc_gamma0_to_sigma0, phase_screen_raster,
                az_time_correction,
                slant_range_correction, input_rtc, output_rtc,
                input_layover_shadow_mask_raster, sub_swaths,
                out_mask, 
                geocode_memory_mode, min_block_size, max_block_size,
                dem_interp_method);
    else if (flag_run_geocode_interp)
        geocodeInterp<float>(radar_grid, input_raster, output_raster,
                dem_raster, flag_apply_rtc, flag_az_baseband_doppler, flatten,
                input_terrain_radiometry, output_terrain_radiometry,
                rtc_min_value_db, rtc_geogrid_upsampling, rtc_algorithm,
                rtc_area_beta_mode,
                abs_cal_factor, clip_min, clip_max, out_geo_rdr, out_geo_dem,
                out_geo_rtc,  out_geo_rtc_gamma0_to_sigma0,
                phase_screen_raster, az_time_correction,
                slant_range_correction, input_rtc, output_rtc,
                input_layover_shadow_mask_raster, sub_swaths,
                out_mask,
                geocode_memory_mode, min_block_size, max_block_size,
                dem_interp_method);
    else if (!flag_complex_to_real)
        geocodeAreaProj<T>(radar_grid, input_raster, output_raster, dem_raster,
                geogrid_upsampling, flag_upsample_radar_grid, flag_apply_rtc,
                input_terrain_radiometry, output_terrain_radiometry,
                rtc_min_value_db, rtc_geogrid_upsampling, rtc_algorithm,
                rtc_area_beta_mode,
                abs_cal_factor, clip_min, clip_max, min_nlooks,
                radar_grid_nlooks, out_off_diag_terms, out_geo_rdr, out_geo_dem,
                out_geo_nlooks, out_geo_rtc,  out_geo_rtc_gamma0_to_sigma0,
                az_time_correction, slant_range_correction,
                input_rtc, output_rtc,
                input_layover_shadow_mask_raster, sub_swaths,
                out_mask,
                geocode_memory_mode, min_block_size, max_block_size,
                dem_interp_method);
    else if (std::is_same<T, double>::value ||
             std::is_same<T, std::complex<double>>::value)
        geocodeAreaProj<double>(radar_grid, input_raster, output_raster,
                dem_raster, geogrid_upsampling, flag_upsample_radar_grid,
                flag_apply_rtc, input_terrain_radiometry,
                output_terrain_radiometry, rtc_min_value_db,
                rtc_geogrid_upsampling, rtc_algorithm, rtc_area_beta_mode,
                abs_cal_factor, clip_min,
                clip_max, min_nlooks, radar_grid_nlooks, out_off_diag_terms,
                out_geo_rdr, out_geo_dem, out_geo_nlooks, out_geo_rtc,
                out_geo_rtc_gamma0_to_sigma0, az_time_correction,
                slant_range_correction, input_rtc,
                output_rtc, input_layover_shadow_mask_raster, sub_swaths,
                out_mask, geocode_memory_mode,
                min_block_size, max_block_size, dem_interp_method);
    else
        geocodeAreaProj<float>(radar_grid, input_raster, output_raster,
                dem_raster, geogrid_upsampling, flag_upsample_radar_grid,
                flag_apply_rtc, input_terrain_radiometry,
                output_terrain_radiometry, rtc_min_value_db,
                rtc_geogrid_upsampling, rtc_algorithm, rtc_area_beta_mode,
                abs_cal_factor, clip_min,
                clip_max, min_nlooks, radar_grid_nlooks, out_off_diag_terms,
                out_geo_rdr, out_geo_dem, out_geo_nlooks, out_geo_rtc,
                out_geo_rtc_gamma0_to_sigma0, az_time_correction,
                slant_range_correction, input_rtc,
                output_rtc, input_layover_shadow_mask_raster, sub_swaths,
                out_mask, geocode_memory_mode,
                min_block_size, max_block_size, dem_interp_method);
}

template<class T>
template<class T_out>
void Geocode<T>::geocodeInterp(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& inputRaster, isce3::io::Raster& outputRaster,
        isce3::io::Raster& demRaster, bool flag_apply_rtc,
        bool flag_az_baseband_doppler, bool flatten,
        isce3::geometry::rtcInputTerrainRadiometry input_terrain_radiometry,
        isce3::geometry::rtcOutputTerrainRadiometry output_terrain_radiometry,
        float rtc_min_value_db, double rtc_geogrid_upsampling,
        isce3::geometry::rtcAlgorithm rtc_algorithm,
        isce3::geometry::rtcAreaBetaMode rtc_area_beta_mode,
        double abs_cal_factor,
        float clip_min, float clip_max, isce3::io::Raster* out_geo_rdr,
        isce3::io::Raster* out_geo_dem, isce3::io::Raster* out_geo_rtc,
        isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0,
        isce3::io::Raster* phase_screen_raster,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        isce3::io::Raster* input_rtc,
        isce3::io::Raster* output_rtc,
        isce3::io::Raster* input_layover_shadow_mask_raster,
        isce3::product::SubSwaths* sub_swaths,
        isce3::io::Raster* out_mask,
        isce3::core::GeocodeMemoryMode geocode_memory_mode, const long long min_block_size,
        const long long max_block_size,
        isce3::core::dataInterpMethod dem_interp_method)
{
    pyre::journal::info_t info("isce.geocode.GeocodeCov.geocodeInterp");
    pyre::journal::warning_t warning("isce.geocode.GeocodeCov.geocodeInterp");
    auto start_time = std::chrono::high_resolution_clock::now();

    isce3::product::GeoGridParameters geogrid(_geoGridStartX, _geoGridStartY,
            _geoGridSpacingX, _geoGridSpacingY, _geoGridWidth, _geoGridLength,
            _epsgOut);

    info << "geo2rdr threshold: " << _threshold << pyre::journal::newline;
    info << "geo2rdr numiter: " << _numiter << pyre::journal::newline;
    info << "baseband azimuth spectrum (0:false, 1:true): "
         << flag_az_baseband_doppler << pyre::journal::newline;
    info << "flatten phase (0:false, 1:true): " << flatten
         << pyre::journal::newline;

    info << "remove phase screen (0: false, 1: true): "
            << std::to_string(phase_screen_raster != nullptr)
            << pyre::journal::newline;
    info << "apply azimuth offset (0: false, 1: true): "
            << std::to_string(az_time_correction.haveData())
            << pyre::journal::newline;
    info << "apply range offset (0: false, 1: true): "
            << std::to_string(slant_range_correction.haveData())
            << pyre::journal::newline;

    // number of bands in the input raster
    int nbands = inputRaster.numBands();
    info << "nbands: " << nbands << pyre::journal::newline;
    // create projection based on _epsg code
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(_epsgOut));

    // make sure int type rasters only used with nearest neighbor
    for (int band = 0; band < nbands; ++band)
    {
        const auto dtype = inputRaster.dtype(band + 1);
        if ((dtype == GDT_Byte || dtype == GDT_UInt32)
                && _data_interp_method != isce3::core::NEAREST_METHOD)
        {
            pyre::journal::error_t error(
                    "isce3.geocode.GeocodeCov.geocodeInterp");
            error << "int type of raster can only use nearest neighbor interp"
                << pyre::journal::endl;
            std::string err_str {
                "int type of raster can only use nearest neighbor interp"};
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), err_str);
        }
    }

    isce3::core::Matrix<uint8_t> input_layover_shadow_mask;
    if (input_layover_shadow_mask_raster != nullptr) {
        info << "input layover/shadow mask provided: True" <<
            pyre::journal::newline;
        _validateInputLayoverShadowMaskRaster(
            input_layover_shadow_mask_raster, radar_grid);

        input_layover_shadow_mask.resize(
                radar_grid.length(), radar_grid.width());
        input_layover_shadow_mask_raster->getBlock(
            input_layover_shadow_mask.data(), 0, 0,
            radar_grid.width(), radar_grid.length(), 1);
    }

    // create data interpolator
    std::unique_ptr<isce3::core::Interpolator<T_out>> interp {
            isce3::core::createInterpolator<T_out>(_data_interp_method)};

    // read phase screen
    isce3::core::Matrix<float> phase_screen_array;
    if (phase_screen_raster != nullptr) {
        phase_screen_array.resize(radar_grid.length(), radar_grid.width());
        phase_screen_raster->getBlock(phase_screen_array.data(), 0, 0,
                radar_grid.width(), radar_grid.length(), 1);
    }

    if (!std::isnan(clip_min))
        info << "clip min: " << clip_min << pyre::journal::newline;

    if (!std::isnan(clip_max))
        info << "clip max: " << clip_max << pyre::journal::newline;

    // RTC
    double rtc_min_value = 0;
    if (!std::isnan(rtc_min_value_db) && flag_apply_rtc) {
        rtc_min_value = std::pow(10, (rtc_min_value_db / 10));
        info << "RTC min. value: " << rtc_min_value_db
             << " [dB] = " << rtc_min_value << pyre::journal::newline;
    }

    if (abs_cal_factor != 1)
        info << "absolute calibration factor: " << abs_cal_factor
             << pyre::journal::newline;

    isce3::io::Raster* rtc_raster;
    isce3::io::Raster* rtc_sigma0_raster = nullptr;
    std::unique_ptr<isce3::io::Raster> rtc_raster_unique_ptr;
    std::unique_ptr<isce3::io::Raster> rtc_raster_sigma0_unique_ptr;
    isce3::core::Matrix<float> rtc_area_array, rtc_area_sigma0_array;

    info << "flag_apply_rtc (0:false, 1:true): " << flag_apply_rtc
         << pyre::journal::newline;

    if (flag_apply_rtc) {
        std::string input_terrain_radiometry_str =
                get_input_terrain_radiometry_str(input_terrain_radiometry);
        info << "input terrain radiometry: " << input_terrain_radiometry_str
             << pyre::journal::newline;

        if (input_rtc == nullptr) {

            info << "calling RTC (from geocode)..." << pyre::journal::newline;

            // if RTC (area factor) raster does not needed to be saved,
            // initialize it as a GDAL memory virtual file
            if (output_rtc == nullptr) {
                std::string vsimem_ref = (
                    "/vsimem/" + getTempString("geocode_cov_interp_rtc"));
                rtc_raster_unique_ptr = std::make_unique<isce3::io::Raster>(
                        vsimem_ref, radar_grid.width(),
                        radar_grid.length(), 1, GDT_Float32, "ENVI");
                rtc_raster = rtc_raster_unique_ptr.get();
            }

            // Otherwise, copies the pointer to the output RTC file
            else
                rtc_raster = output_rtc;

            isce3::geometry::rtcAreaMode rtc_area_mode =
                    isce3::geometry::rtcAreaMode::AREA_FACTOR;

            if (std::isnan(rtc_geogrid_upsampling))
                rtc_geogrid_upsampling = 1;

            isce3::core::MemoryModeBlocksY rtc_memory_mode;
            if (geocode_memory_mode == isce3::core::GeocodeMemoryMode::Auto)
                rtc_memory_mode = isce3::core::MemoryModeBlocksY::AutoBlocksY;
            else if (geocode_memory_mode == isce3::core::GeocodeMemoryMode::SingleBlock)
                rtc_memory_mode = isce3::core::MemoryModeBlocksY::SingleBlockY;
            else
                rtc_memory_mode = isce3::core::MemoryModeBlocksY::MultipleBlocksY;

            if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
                std::string vsimem_ref = (
                    "/vsimem/" + getTempString("geocode_cov_areaproj_rtc_sigma0"));
                rtc_raster_sigma0_unique_ptr = 
                    std::make_unique<isce3::io::Raster>(
                        vsimem_ref, radar_grid.width(),
                        radar_grid.length(), 1, GDT_Float32, "ENVI");
                rtc_sigma0_raster = 
                    rtc_raster_sigma0_unique_ptr.get();
            }

            isce3::io::Raster* out_geo_rdr = nullptr;
            isce3::io::Raster* out_geo_grid = nullptr;

            computeRtc(demRaster, *rtc_raster, radar_grid, _orbit, _doppler,
                    _geoGridStartY, _geoGridSpacingY, _geoGridStartX,
                    _geoGridSpacingX, _geoGridLength, _geoGridWidth, _epsgOut,
                    input_terrain_radiometry, output_terrain_radiometry,
                    rtc_area_mode, rtc_algorithm, rtc_area_beta_mode,
                    rtc_geogrid_upsampling, rtc_min_value_db,
                    out_geo_rdr, out_geo_grid,
                    rtc_sigma0_raster, rtc_memory_mode,
                    dem_interp_method, _threshold,
                    _numiter, 1.0e-8, min_block_size, max_block_size);

        } else {
            info << "reading pre-computed RTC..." << pyre::journal::newline;
            rtc_raster = input_rtc;
        }

        rtc_area_array.resize(radar_grid.length(), radar_grid.width());
        rtc_raster->getBlock(rtc_area_array.data(), 0, 0, radar_grid.width(),
                radar_grid.length(), 1);

        if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
            rtc_area_sigma0_array.resize(radar_grid.length(),
                                         radar_grid.width());
            rtc_sigma0_raster->getBlock(rtc_area_sigma0_array.data(),
                                        0, 0, radar_grid.width(),
                                        radar_grid.length(), 1);
        }
    }

    geogrid.print();
    _print_parameters(info, geocode_memory_mode, min_block_size,
                      max_block_size);

    int nBlocks, block_length;

    if (geocode_memory_mode == isce3::core::GeocodeMemoryMode::SingleBlock) {
        nBlocks = 1;
        block_length = geogrid.length();
    } else {

        if (geocode_memory_mode == 
                isce3::core::GeocodeMemoryMode::BlocksGeogridAndRadarGrid) {
            warning << "WARNING the geocode memory mode"
                 << " BlocksGeogridAndRadarGrid is not available"
                 << " for geocoding with interpolation. Using"
                 << " memory mode BlocksGeogrid instead."
                 << pyre::journal::newline;
        }

        isce3::core::getBlockProcessingParametersY(
            geogrid.length(), geogrid.width(), nbands, sizeof(T),
            &info, &block_length, &nBlocks, min_block_size, max_block_size);
    } 

    info << "number of blocks: " << nBlocks << pyre::journal::newline;
    info << "block length: " << block_length << pyre::journal::newline;
    info << pyre::journal::newline;

    info << "starting geocoding" << pyre::journal::endl;
    // loop over the blocks of the geocoded Grid
    for (int block = 0; block < nBlocks; ++block) {
        info << "block: " << block << pyre::journal::endl;
        // Get block extents (of the geocoded grid)
        int lineStart = block * block_length;
        int geoBlockLength = block_length;
        if (block == (nBlocks - 1)) {
            geoBlockLength = geogrid.length() - lineStart;
        }

        int blockSize = geoBlockLength * geogrid.width();

        isce3::core::Matrix<float> out_geo_rdr_a;
        isce3::core::Matrix<float> out_geo_rdr_r;
        if (out_geo_rdr != nullptr) {
            out_geo_rdr_a.resize(geoBlockLength, geogrid.width());
            out_geo_rdr_r.resize(geoBlockLength, geogrid.width());
            out_geo_rdr_a.fill(std::numeric_limits<float>::quiet_NaN());
            out_geo_rdr_r.fill(std::numeric_limits<float>::quiet_NaN());
        }

        isce3::core::Matrix<float> out_geo_dem_array;
        if (out_geo_dem != nullptr) {
            out_geo_dem_array.resize(geoBlockLength, geogrid.width());
            out_geo_dem_array.fill(std::numeric_limits<float>::quiet_NaN());
        }

       // load a block of DEM for the current geocoded grid with a margin of
        // 50 DEM pixels
        int dem_margin_in_pixels = 50;
        isce3::geometry::DEMInterpolator demInterp =
            isce3::geometry::DEMRasterToInterpolator(
                demRaster, geogrid, lineStart, geoBlockLength, geogrid.width(),
                dem_margin_in_pixels, dem_interp_method);

        // X and Y indices (in the radar coordinates) for the
        // geocoded pixels (after geo2rdr computation)
        std::valarray<double> radarX(blockSize);
        std::valarray<double> radarY(blockSize);

        int azimuthFirstLine = radar_grid.length() - 1;
        int azimuthLastLine = 0;
        int rangeFirstPixel = radar_grid.width() - 1;
        int rangeLastPixel = 0;

        // Loop over lines, samples of the output grid
#pragma omp parallel for reduction(                                            \
        min                                                                    \
        : azimuthFirstLine, rangeFirstPixel)                         \
        reduction(max                                                          \
                  : azimuthLastLine, rangeLastPixel)

        for (size_t kk = 0; kk < geoBlockLength * geogrid.width(); ++kk) {

            size_t blockLine = kk / geogrid.width();
            size_t pixel = kk % geogrid.width();

            // Global line index
            const int line = lineStart + blockLine;

            // y coordinate in the out put grid
            double y = geogrid.startY() + geogrid.spacingY() * (0.5 + line);

            // x in the output geocoded Grid
            double x = geogrid.startX() + geogrid.spacingX() * (0.5 + pixel);

            // compute the azimuth time and slant range for the
            // x,y coordinates in the output grid
            double aztime, srange;
            float dem_value;

            aztime = radar_grid.sensingMid();
            int converged = _geo2rdr(radar_grid, x, y, aztime, srange,
                    demInterp, proj.get(), dem_value);

            // (optional arg) save interpolated DEM element
            if (out_geo_dem != nullptr) {
#pragma omp atomic write
                out_geo_dem_array(blockLine, pixel) = dem_value;
            }

            if (!converged)
                continue;

            // apply timing corrections
            if (az_time_correction.contains(aztime, srange)) {
                const auto aztimeCor = az_time_correction.eval(aztime,
                                                                srange);
                aztime += aztimeCor;
            }

            if (slant_range_correction.contains(aztime, srange)) {
                const auto srangeCor = slant_range_correction.eval(aztime,
                                                                    srange);
                srange += srangeCor;
            }

            // check if az time and slant within radar grid
            if (!radar_grid.contains(aztime, srange)
                    || !_nativeDoppler.contains(aztime, srange))
                continue;

            // get the row and column index in the radar grid
            double rdrY = ((aztime - radar_grid.sensingStart()) /
                           radar_grid.azimuthTimeInterval());

            double rdrX = ((srange - radar_grid.startingRange()) /
                           radar_grid.rangePixelSpacing());

            // (optional arg) save rdr pos element
            if (out_geo_rdr != nullptr) {
#pragma omp atomic write
                out_geo_rdr_a(blockLine, pixel) = rdrY;
#pragma omp atomic write
                out_geo_rdr_r(blockLine, pixel) = rdrX;
            }

            if (rdrY < 0 || rdrX < 0 || rdrY >= radar_grid.length() ||
                    rdrX >= radar_grid.width())
                continue;

            azimuthFirstLine = std::min(
                    azimuthFirstLine, static_cast<int>(std::floor(rdrY)));
            azimuthLastLine = std::max(azimuthLastLine,
                    static_cast<int>(std::ceil(rdrY) - 1));
            rangeFirstPixel = std::min(
                    rangeFirstPixel, static_cast<int>(std::floor(rdrX)));
            rangeLastPixel = std::max(
                    rangeLastPixel, static_cast<int>(std::ceil(rdrX) - 1));

            // store the adjusted X and Y indices
            radarX[blockLine * geogrid.width() + pixel] = rdrX;
            radarY[blockLine * geogrid.width() + pixel] = rdrY;

        } // end loops over lines and pixel of output grid

        // (optional arg) flush rdr position values
        if (out_geo_rdr != nullptr)
#pragma omp critical
        {
            out_geo_rdr->setBlock(out_geo_rdr_a.data(), 0, lineStart,
                    geogrid.width(), geoBlockLength, 1);
            out_geo_rdr->setBlock(out_geo_rdr_r.data(), 0, lineStart,
                    geogrid.width(), geoBlockLength, 2);
        }

        // (optional arg) flush interpolated DEM values
        if (out_geo_dem != nullptr)
#pragma omp critical
        {
            out_geo_dem->setBlock(out_geo_dem_array.data(), 0, lineStart,
                    geogrid.width(), geoBlockLength, 1);
        }

        // Add extra margin for interpolation. We set it to 5 pixels marging
        // considering SINC interpolation that requires 9 pixels
        int interp_margin = 5;
        azimuthFirstLine = std::max(azimuthFirstLine - interp_margin, 0);
        rangeFirstPixel = std::max(rangeFirstPixel - interp_margin, 0);

        azimuthLastLine = std::min(azimuthLastLine + interp_margin,
                                   static_cast<int>(radar_grid.length() - 1));
        rangeLastPixel = std::min(rangeLastPixel + interp_margin,
                                  static_cast<int>(radar_grid.width() - 1));

        // set NaN values according to T_out, i.e. real (NaN) or complex (NaN,
        // NaN)
        using T_out_real = typename isce3::real<T_out>::type;
        T_out nan_t_out = 0;
        nan_t_out *= std::numeric_limits<T_out_real>::quiet_NaN();

        // define the geo-block matrix based on the raster bands data type
        isce3::core::Matrix<T_out> geoDataBlock(
                geoBlockLength, geogrid.width());
        geoDataBlock.fill(nan_t_out);

        // if invalid, fill all bands with NaNs and continue to the next block
        if (azimuthFirstLine > azimuthLastLine ||
                rangeFirstPixel > rangeLastPixel) {
            for (int band = 0; band < nbands; ++band) {
                outputRaster.setBlock(geoDataBlock.data(), 0, lineStart,
                        geogrid.width(), geoBlockLength, band + 1);
            }
            continue;
        }

        // shape of the required block of data in the radar coordinates
        int rdrBlockLength = azimuthLastLine - azimuthFirstLine + 1;
        int rdrBlockWidth = rangeLastPixel - rangeFirstPixel + 1;

        // define the radar-block matrix based on the rasterbands data type
        isce3::core::Matrix<T_out> rdrDataBlock(rdrBlockLength, rdrBlockWidth);
        rdrDataBlock.fill(nan_t_out);

        // for each band in the input:
        for (int band = 0; band < nbands; ++band) {

            // if complex to real
            if ((std::is_same<T, std::complex<float>>::value ||
                        std::is_same<T, std::complex<double>>::value) &&
                    (std::is_same<T_out, float>::value ||
                            std::is_same<T_out, double>::value)) {
                isce3::core::Matrix<T> rdrDataBlockTemp(
                        rdrBlockLength, rdrBlockWidth);
                inputRaster.getBlock(rdrDataBlockTemp.data(), rangeFirstPixel,
                        azimuthFirstLine, rdrBlockWidth, rdrBlockLength,
                        band + 1);
                if (flag_az_baseband_doppler) {

                    // baseband the SLC in the radar grid
                    const double blockStartingRange =
                            radar_grid.startingRange() +
                            rangeFirstPixel * radar_grid.rangePixelSpacing();
                    const double blockSensingStart =
                            radar_grid.sensingStart() +
                            azimuthFirstLine / radar_grid.prf();

                    _baseband(rdrDataBlockTemp, blockStartingRange,
                            blockSensingStart, radar_grid.rangePixelSpacing(),
                            radar_grid.prf(), _nativeDoppler);
                }
                for (int i = 0; i < rdrBlockLength; ++i)
                    for (int j = 0; j < rdrBlockWidth; ++j) {
                        T_out output_value;
                        _convertToOutputType(
                                rdrDataBlockTemp(i, j), output_value);
                        rdrDataBlock(i, j) = output_value;
                    }
            }
            // otherwise
            else {
                inputRaster.getBlock(rdrDataBlock.data(), rangeFirstPixel,
                        azimuthFirstLine, rdrBlockWidth, rdrBlockLength,
                        band + 1);
                if (flag_az_baseband_doppler) {

                    // baseband the SLC in the radar grid
                    const double blockStartingRange =
                            radar_grid.startingRange() +
                            rangeFirstPixel * radar_grid.rangePixelSpacing();
                    const double blockSensingStart =
                            radar_grid.sensingStart() +
                            azimuthFirstLine / radar_grid.prf();

                    _baseband(rdrDataBlock, blockStartingRange,
                            blockSensingStart, radar_grid.rangePixelSpacing(),
                            radar_grid.prf(), _nativeDoppler);
                }
            }

            // (optional arg) if band == 0, populate RTC array
            isce3::io::Raster* out_geo_rtc_band = nullptr;
            isce3::core::Matrix<float> out_geo_rtc_array;
            if (out_geo_rtc != nullptr && band == 0) {
                out_geo_rtc_band = out_geo_rtc;
                out_geo_rtc_array.resize(geoBlockLength, geogrid.width());
                out_geo_rtc_array.fill(std::numeric_limits<float>::quiet_NaN());
            }

            // (optional arg) if band == 0, populate RTC array
            isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0_band = nullptr;
            isce3::core::Matrix<float> out_geo_rtc_gamma0_to_sigma0_array;
            if (out_geo_rtc_gamma0_to_sigma0 != nullptr && band == 0) {
                out_geo_rtc_gamma0_to_sigma0_band = out_geo_rtc_gamma0_to_sigma0;
                out_geo_rtc_gamma0_to_sigma0_array.resize(geoBlockLength, geogrid.width());
                out_geo_rtc_gamma0_to_sigma0_array.fill(std::numeric_limits<float>::quiet_NaN());
            }

            isce3::core::Matrix<short> out_mask_array;
            if (out_mask != nullptr) {
                out_mask_array.resize(geoBlockLength, geogrid.width());
                out_mask_array.fill(0);
            }
 
            _interpolate(rdrDataBlock, geoDataBlock, radarX, radarY,
                    rdrBlockWidth, rdrBlockLength, azimuthFirstLine,
                    rangeFirstPixel, interp.get(), radar_grid,
                    flag_az_baseband_doppler, flatten, phase_screen_raster,
                    phase_screen_array, abs_cal_factor, clip_min, clip_max,
                    flag_apply_rtc, rtc_area_array, rtc_area_sigma0_array, out_geo_rtc_band,
                    out_geo_rtc_array, out_geo_rtc_gamma0_to_sigma0_band,
                    out_geo_rtc_gamma0_to_sigma0_array,
                    input_layover_shadow_mask_raster,
                    input_layover_shadow_mask, sub_swaths,
                    out_mask, out_mask_array);

            // flush optional layers
            if (out_geo_rtc_band != nullptr && band == 0) {
                out_geo_rtc->setBlock(out_geo_rtc_array.data(), 0, lineStart,
                        geogrid.width(), geoBlockLength, 1);
            }
            if (out_geo_rtc_gamma0_to_sigma0_band != nullptr && band == 0) {
                out_geo_rtc_gamma0_to_sigma0->setBlock(
                    out_geo_rtc_gamma0_to_sigma0_array.data(), 0, lineStart,
                    geogrid.width(), geoBlockLength, 1);
            }
            if (out_mask != nullptr && band == nbands - 1) {
                out_mask->setBlock(out_mask_array.data(), 0,
                    lineStart, geogrid.width(), geoBlockLength, 1);
            }

            outputRaster.setBlock(geoDataBlock.data(), 0, lineStart,
                    geogrid.width(), geoBlockLength, band + 1);

        }
    } // end loop over block of output grid

    double geotransform[] = {geogrid.startX(), geogrid.spacingX(), 0,
            geogrid.startY(), 0, geogrid.spacingY()};
    if (geogrid.spacingY() > 0) {
        geotransform[3] =
                geogrid.startY() + geogrid.length() * geogrid.spacingY();
        geotransform[5] = -geogrid.spacingY();
    }

    outputRaster.setGeoTransform(geotransform);
    outputRaster.setEPSG(geogrid.epsg());

    if (out_geo_rdr != nullptr) {
        out_geo_rdr->setGeoTransform(geotransform);
        out_geo_rdr->setEPSG(geogrid.epsg());
    }

    if (out_geo_dem != nullptr) {
        out_geo_dem->setGeoTransform(geotransform);
        out_geo_dem->setEPSG(geogrid.epsg());
    }

    if (out_geo_rtc != nullptr) {
        out_geo_rtc->setGeoTransform(geotransform);
        out_geo_rtc->setEPSG(geogrid.epsg());
    }

    if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
        out_geo_rtc_gamma0_to_sigma0->setGeoTransform(geotransform);
        out_geo_rtc_gamma0_to_sigma0->setEPSG(geogrid.epsg());
    }

    if (out_mask != nullptr) {
        out_mask->setGeoTransform(geotransform);
        out_mask->setEPSG(geogrid.epsg());
    }

    auto elapsed_time_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    float elapsed_time = ((float) elapsed_time_milliseconds.count()) / 1e3;
    info << "elapsed time (GEO-IN) [s]: " << elapsed_time << pyre::journal::endl;
}

template<class T>
template<class T_out>
inline void Geocode<T>::_interpolate(
        const isce3::core::Matrix<T_out>& rdrDataBlock,
        isce3::core::Matrix<T_out>& geoDataBlock,
        const std::valarray<double>& radarX,
        const std::valarray<double>& radarY, const int radarBlockWidth,
        const int radarBlockLength, const int azimuthFirstLine,
        const int rangeFirstPixel,
        const isce3::core::Interpolator<T_out>* interp,
        const isce3::product::RadarGridParameters& radar_grid,
        const bool flag_az_baseband_doppler, const bool flatten,
        isce3::io::Raster* phase_screen_raster,
        isce3::core::Matrix<float>& phase_screen_array, double abs_cal_factor,
        float clip_min, float clip_max, bool flag_apply_rtc,
        const isce3::core::Matrix<float>& rtc_area,
        const isce3::core::Matrix<float>& rtc_area_sigma,
        isce3::io::Raster* out_geo_rtc,
        isce3::core::Matrix<float>& out_geo_rtc_array,
        isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0,
        isce3::core::Matrix<float>& out_geo_rtc_gamma0_to_sigma0_array,
        isce3::io::Raster* input_layover_shadow_mask_raster,
        isce3::core::Matrix<uint8_t>& input_layover_shadow_mask_array,
        isce3::product::SubSwaths * sub_swaths,
        isce3::io::Raster* out_mask,
        isce3::core::Matrix<short>& out_mask_array)
{

    using isce3::math::complex_operations::operator*;

    size_t length = geoDataBlock.length();
    size_t width = geoDataBlock.width();
    // Add extra margin for interpolation
    int interp_margin = 5;

    double offsetY =
            azimuthFirstLine / radar_grid.prf() + radar_grid.sensingStart();
    double offsetX = rangeFirstPixel * radar_grid.rangePixelSpacing() +
                     radar_grid.startingRange();

#pragma omp parallel for
    for (size_t kk = 0; kk < length * width; ++kk) {

        size_t i = kk / width;
        size_t j = kk % width;

        // adjust the row and column indicies for the current block,
        // i.e., moving the origin to the top-left of this radar block.
        double rdrY = radarY[i * width + j] - azimuthFirstLine;
        double rdrX = radarX[i * width + j] - rangeFirstPixel;

        if (rdrX < interp_margin || rdrY < interp_margin ||
                rdrX >= (radarBlockWidth - interp_margin) ||
                rdrY >= (radarBlockLength - interp_margin)) {

            // set NaN values according to T_out, i.e. real (NaN) or complex
            // (NaN, NaN)
            using T_out_real = typename isce3::real<T_out>::type;
            geoDataBlock(i, j) *= std::numeric_limits<T_out_real>::quiet_NaN();
            if (flag_apply_rtc && out_geo_rtc != nullptr) {
                out_geo_rtc_array(i, j) = std::numeric_limits<float>::quiet_NaN();
            }
            if (flag_apply_rtc && out_geo_rtc_gamma0_to_sigma0 != nullptr) {
                out_geo_rtc_gamma0_to_sigma0_array(i, j) =
                    std::numeric_limits<float>::quiet_NaN();
            }
            if (out_mask != nullptr) {
                out_mask_array(i, j) = 0;
            }
            continue;
        }

        int rdr_y_rslc = std::floor(rdrY + azimuthFirstLine);
        int rdr_x_rslc = std::floor(rdrX + rangeFirstPixel);

        short sample_sub_swath_center = 1;
        if (sub_swaths != nullptr) {
            bool flag_skip = false;
            for (int yy = -interp_margin; yy <= interp_margin; ++yy) {
                for (int xx = -interp_margin; xx <= interp_margin; ++xx) {
                    short sample_sub_swath = sub_swaths->getSampleSubSwath(
                        rdr_y_rslc + yy, rdr_x_rslc + xx);
                    if (sample_sub_swath == 0) {
                        // set NaN values according to T_out, i.e. real (NaN)
                        // or complex (NaN, NaN)
                        using T_out_real = typename isce3::real<T_out>::type;
                        geoDataBlock(i, j) *= 
                            std::numeric_limits<T_out_real>::quiet_NaN();
                        if (flag_apply_rtc && out_geo_rtc != nullptr) {
                            out_geo_rtc_array(i, j) =
                                std::numeric_limits<float>::quiet_NaN();
                        }
                        if (flag_apply_rtc &&
                                out_geo_rtc_gamma0_to_sigma0 != nullptr) {
                            out_geo_rtc_gamma0_to_sigma0_array(i, j) =
                                std::numeric_limits<float>::quiet_NaN();
                        }
                        if (out_mask != nullptr) {
                            out_mask_array(i, j) = 0;
                        }
                        flag_skip = true;
                        break;
                    }
                    if (yy == 0 && xx == 0) {
                        sample_sub_swath_center = sample_sub_swath;
                    }
 
                }
                if (flag_skip) {
                    break;
                }
            }
            if (flag_skip) {
                continue;
            }
        }
        if (out_mask != nullptr) {
            out_mask_array(i, j) = sample_sub_swath_center;
        }

        /* 
        check within the interpolation kernel (approximated by `interp_margin`)
        if any of the samples is marked as shadow or layover-and-shadow
        in which case we skip to the next position, i.e., we "break" the 
        2 inner for-loop bellow (vars: yy and xx) and "continue" from the parent
        for-loop (var: kk) above.
        */
        if (input_layover_shadow_mask_raster != nullptr) {
            bool flag_skip = false;
            for (int yy = -interp_margin; yy <= interp_margin; ++yy) {
                for (int xx = -interp_margin; xx <= interp_margin; ++xx) {
                    const uint8_t input_layover_shadow_value = \
                            input_layover_shadow_mask_array(rdr_y_rslc + yy,
                                                      rdr_x_rslc + xx);
                    if (input_layover_shadow_value == SHADOW_VALUE ||
                            input_layover_shadow_value == LAYOVER_AND_SHADOW_VALUE) {
                        flag_skip = true;
                        break;
                    }
                }
                if (flag_skip) {
                    break;
                }
            }
            if (flag_skip) {
                continue;
            }
        }

        // Interpolate chip
        T_out val = interp->interpolate(rdrX, rdrY, rdrDataBlock);

        if (!isnan(abs_cal_factor) && abs_cal_factor != 1)
            val *= abs_cal_factor;

        if (flag_apply_rtc) {
            float rtc_value =
                    rtc_area(int(rdrY + azimuthFirstLine),
                             int(rdrX + rangeFirstPixel));
            val /= std::sqrt(rtc_value);
            if (out_geo_rtc != nullptr) {
                out_geo_rtc_array(i, j) = rtc_value;
            }

            if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
                /*
                The RTC area normalization factor (ANF) gamma0 to sigma0
                is computed from the RTC ANF gamma0 to beta0 (or
                sigma0-ellipsoid) `rtc_value` divided by the RTC ANF sigma0
                to beta0 `rtc_sigma0`
                */
                float rtc_sigma0 = rtc_area_sigma(int(rdrY + azimuthFirstLine),
                                                int(rdrX + rangeFirstPixel));
                const double rtc_gamma0_to_sigma0 = rtc_value / rtc_sigma0;
                out_geo_rtc_gamma0_to_sigma0_array(i, j) = rtc_gamma0_to_sigma0;
            }
        }

        // clip min (complex)
        if (!std::isnan(clip_min) && std::abs(val) < clip_min &&
                isce3::is_complex<T_out>())
            val = val * clip_min / std::abs(val);

        // clip min (real)
        else if (!std::isnan(clip_min) && std::abs(val) < clip_min)
            val = clip_min;

        // clip max (complex)
        if (!std::isnan(clip_max) && std::abs(val) > clip_max &&
                isce3::is_complex<T_out>())
            val = val * clip_max / std::abs(val);

        // clip max (real)
        else if (!std::isnan(clip_max) && std::abs(val) > clip_max)
            val = clip_max;

        if (std::is_same<T_out, float>::value ||
                std::is_same<T_out, double>::value ||
                (!flag_az_baseband_doppler && !flatten)) {
            geoDataBlock(i, j) = val;
            continue;
        }

        double aztime = rdrY / radar_grid.prf() + offsetY;
        double srange = rdrX * radar_grid.rangePixelSpacing() + offsetX;

        // doppler to be added back after interpolation
        double phase = 0;

        if (flag_az_baseband_doppler) {
            phase += _nativeDoppler.eval(aztime, srange) * 2 * M_PI * aztime;
        }

        if (flatten) {
            phase += (4.0 * (M_PI / radar_grid.wavelength())) * srange;
        }

        if (phase_screen_raster != nullptr) {
            phase -= phase_screen_array(
                    int(rdrY + azimuthFirstLine), int(rdrX + rangeFirstPixel));
        }

        T_out cpxPhase;
        using T_real = typename isce3::real<T_out>::type;
        _convertToOutputType(
                std::complex<T_real>(std::cos(phase), std::sin(phase)),
                cpxPhase);
        geoDataBlock(i, j) = val * cpxPhase;

    } // end for
}

/*
_baseband() moves the azimuth spectrum to baseband using the Doppler LUT.
This is only useful applicable for complex images. For real images,
the function does nothing. The "dummy function" bellow overloads
 _baseband() for real images.
*/
template<class T>
template<class T2>
void Geocode<T>::_baseband(isce3::core::Matrix<T2>& data,
        const double starting_range, const double sensing_start,
        const double range_pixel_spacing, const double prf,
        const isce3::core::LUT2d<double>& doppler_lut)
{
    // tells the compiler to ignore these unused variables:
    (void) data;
    (void) starting_range;
    (void) sensing_start;
    (void) range_pixel_spacing;
    (void) prf;
    (void) doppler_lut;
}

template<class T>
template<class T2>
void Geocode<T>::_baseband(isce3::core::Matrix<std::complex<T2>>& data,
        const double starting_range, const double sensing_start,
        const double range_pixel_spacing, const double prf,
        const isce3::core::LUT2d<double>& doppler_lut)
{

    size_t length = data.length();
    size_t width = data.width();

#pragma omp parallel for
    for (size_t kk = 0; kk < length * width; ++kk) {
        size_t line = kk / width;
        size_t col = kk % width;
        const double azimuth_time = sensing_start + line / prf;
        const double slant_range = starting_range + col * range_pixel_spacing;
        const double phase = doppler_lut.eval(azimuth_time, slant_range) * 2 *
                             M_PI * azimuth_time;
        const std::complex<T2> cpx_phase(std::cos(phase), -std::sin(phase));
        data(line, col) *= cpx_phase;
    }
}

template<class T>
int Geocode<T>::_geo2rdr(const isce3::product::RadarGridParameters& radar_grid,
        double x, double y, double& azimuthTime, double& slantRange,
        isce3::geometry::DEMInterpolator& demInterp,
        isce3::core::ProjectionBase* proj, float& dem_value)
{
    // coordinate in the output projection system
    const Vec3 xyz {x, y, 0.0};

    // transform the xyz in the output projection system to llh
    Vec3 llh = proj->inverse(xyz);

    // interpolate the height from the DEM for this pixel
    llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

    // assign interpolated DEM value to returning variable
    dem_value = llh[2];

    // Perform geo->rdr iterations
    int converged = isce3::geometry::geo2rdr(llh, _ellipsoid, _orbit, _doppler,
            azimuthTime, slantRange, radar_grid.wavelength(),
            radar_grid.lookSide(), _threshold, _numiter, 1.0e-8);

    // Check convergence
    if (converged == 0) {
        azimuthTime = std::numeric_limits<double>::quiet_NaN();
        slantRange = std::numeric_limits<double>::quiet_NaN();
    }
    return converged;
}

/*
This function upsamples the complex input by a factor of 2 in the
range domain and converts the complex input to the output that can be either
real or complex.
*/
template<class T, class T_out>
void _processUpsampledBlock(isce3::core::Matrix<T_out>* mat, size_t block,
                            int radar_block_size,
                            isce3::io::Raster& input_raster, size_t xidx,
                            size_t yidx, size_t size_x, size_t size_y,
                            size_t band)
{
    using T_real = typename isce3::real<T>::type;
    size_t this_block_size = radar_block_size;
    if ((block + 1) * radar_block_size > size_y)
        this_block_size = size_y % radar_block_size;
    size_t yidx_block = block * radar_block_size + yidx;

    std::valarray<std::complex<T_real>> refSlcUpsampled(size_x *
                                                        this_block_size);

    /*
    Reads the input raster and upsample the complex array in the X (range)
    direction.
    */

    isce3::signal::upsampleRasterBlockX<T_real>(
            input_raster, refSlcUpsampled, xidx / 2.0, yidx_block,
            size_x / 2.0, this_block_size, band + 1);

    /*
    Iteratively converts input pixel (ptr_1) to output pixel (ptr_2).
    In this case, the input type T is expected to be complex and the output
    type T_out can be real or complex.
    The conversion from complex (e.g. SLC) to real (SAR backscatter)
    in the context of geocoding is considered as the geocoding of
    the covariance matrix (diagonal elements) is done by
    squaring the modulus of the complex input. The conversion between
    variables of same type is considered as regular geocoding and
    no square operation is performed. Both operations are handled by
    the function _convertToOutputType().
    */
    auto ptr_1 = &refSlcUpsampled[0];
    auto ptr_2 = mat->data() + (block * radar_block_size * size_x);
    for (size_t k = 0; k < this_block_size * size_x; ++k) {
        _convertToOutputType(*ptr_1++, *ptr_2++);
    }
}

template<class T, class T_out>
void _getUpsampledBlock(
        std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>>& rdrData,
        isce3::io::Raster& input_raster, size_t xidx, size_t yidx,
        size_t size_x, size_t size_y, bool flag_upsample_radar_grid,
        GeocodeMemoryMode geocode_memory_mode, const long long min_block_size,
        const long long max_block_size, pyre::journal::info_t& info)
{
    int nbands = input_raster.numBands();
    rdrData.reserve(nbands);
    bool flag_parallel_radargrid_read = geocode_memory_mode ==
            GeocodeMemoryMode::BlocksGeogridAndRadarGrid;

    int radargrid_nblocks, radar_block_length;
    const int n_threads_per_radargrid_block = 1;

    for (int band = 0; band < nbands; ++band) {
        if (!flag_parallel_radargrid_read) {
            info << "reading input raster band: " << band + 1
                 << pyre::journal::endl;
        }

        rdrData.emplace_back(
                std::make_unique<isce3::core::Matrix<T_out>>(size_y, size_x));

        if (!flag_upsample_radar_grid && std::is_same<T, T_out>::value &&
                !flag_parallel_radargrid_read) {
            /*
            Enter here if:
                1. No upsampling is required;
                2. No type convertion is required (input and output have same
                   types);
                3. Not parallel (which allows messages to be printed to stdout).
            */

            isce3::core::getBlockProcessingParametersY(size_y, size_x, 1,
                    sizeof(T), nullptr, &radar_block_length, &radargrid_nblocks,
                    min_block_size, max_block_size, n_threads_per_radargrid_block);

            for (size_t block = 0; block < (size_t) radargrid_nblocks;
                 ++block) {

                int this_radar_block_length = radar_block_length;
                if ((block + 1) * radar_block_length > size_y) {
                    this_radar_block_length = size_y % radar_block_length;
                }
                if (radargrid_nblocks > 1) {
                    std::cout << "reading band " << band + 1 << " progress: "
                              << static_cast<int>(
                                         (100.0 * block) / radargrid_nblocks)
                              << "% \r";
                    std::cout.flush();
                }
                auto ptr = rdrData[band]->data();
                input_raster.getBlock(ptr +
                                      block * radar_block_length * size_x,
                                      xidx, block * radar_block_length + yidx, size_x,
                                      this_radar_block_length, band + 1);
            }

            if (radargrid_nblocks > 1) {
                std::cout << "reading band " << band + 1
                          << " progress: 100%" << std::endl;
            }
        }
        else if (!flag_upsample_radar_grid && std::is_same<T, T_out>::value) {
            /*
            Enter here if:
                1. No upsampling is required;
                2. No type convertion is required (input and output have same
                   types);
                3. Is parallel (which does not allow messages to be printed to
                   stdout).
            */
            _Pragma("omp critical")
            {
            input_raster.getBlock(rdrData[band]->data(), xidx, yidx, size_x,
                                  size_y, band + 1);
            }
        }
        else if (!flag_upsample_radar_grid) {
            /*
            Enter here if:
                1. No upsampling is required;
                2. Type convertion is required (input and output have different
                   types).
            */
            isce3::core::Matrix<T> radar_data_in(size_y, size_x);
            if (flag_parallel_radargrid_read) {
                _Pragma("omp critical")
                {
                input_raster.getBlock(radar_data_in.data(), xidx, yidx, size_x,
                                       size_y, band + 1);
                }
            } else {
                input_raster.getBlock(radar_data_in.data(), xidx, yidx, size_x,
                        size_y, band + 1);
            }

            /*
            Iteratively converts input pixel (ptr_1) to output pixel (ptr_2).
            In this case, the input type T (complex) is different than T_out
            (real).
            The conversion from complex (e.g. SLC) to real (SAR backscatter)
            in the context of a covariance matrix (diagonal elements) is done by
            squaring the modulus of the complex input. This operation is handled
            by _convertToOutputType
            */
            auto ptr_1 = radar_data_in.data();
            auto ptr_2 = rdrData[band]->data();
            for (size_t k = 0; k < size_y * size_x; ++k) {
                _convertToOutputType(*ptr_1++, *ptr_2++);
            }
        } else if (flag_upsample_radar_grid && !isce3::is_complex<T>()) {
            /*
            Enter here if:
                1. Upsampling is required;
                2. Input is not complex.
            */
            std::string error_msg = "radar-grid upsampling is only available";
            error_msg += " for complex inputs";
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
        } else {
            /*
            Enter here if:
                1. Upsampling is required;
                2. Input is complex.
            */
            if (geocode_memory_mode == GeocodeMemoryMode::SingleBlock ||
                geocode_memory_mode ==
                        GeocodeMemoryMode::BlocksGeogridAndRadarGrid) {
                radargrid_nblocks = 1;
                radar_block_length = size_y;

            } else {
                isce3::core::getBlockProcessingParametersY(size_y, size_x, 1,
                        sizeof(T), nullptr, &radar_block_length,
                        &radargrid_nblocks, min_block_size, max_block_size,
                        n_threads_per_radargrid_block);
            }
            if (radargrid_nblocks == 1) {
                _processUpsampledBlock<T, T_out>(rdrData[band].get(), 0,
                        radar_block_length, input_raster, xidx, yidx, size_x,
                        size_y, band);
            } else {
                // #pragma omp parallel for schedule(dynamic)
                for (size_t block = 0; block < (size_t) radargrid_nblocks;
                     ++block) {
                    _processUpsampledBlock<T, T_out>(rdrData[band].get(), block,
                            radar_block_length, input_raster, xidx, yidx,
                            size_x, size_y, band);
                }
            }
        }
    }
}

static int _geo2rdrWrapper(const Vec3& inputLLH, const Ellipsoid& ellipsoid,
        const Orbit& orbit, const LUT2d<double>& doppler, double& aztime,
        double& slantRange, double wavelength, LookSide side,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        double threshold, int maxIter, double deltaRange,
        bool flag_edge = true)
{
    int flag_converged;
    for (int i = 0; i <= static_cast<int>(flag_edge); ++i) {
        /*
          Run geo2rdr twice for border edge pixels. This is
          required because initial guesses (a11 and r11)
          are not as good for edge elements. Without it,
          the edge solutions are slightly different than the
          corresponding solutions from single-block processing.
       */
        flag_converged = isce3::geometry::geo2rdr(inputLLH, ellipsoid, orbit,
                doppler, aztime, slantRange, wavelength, side, threshold,
                maxIter, deltaRange);

        if (!flag_converged) {
            return flag_converged;
        }
    }
    // apply timing corrections
    if (az_time_correction.contains(aztime, slantRange)) {
        const auto aztimeCor = az_time_correction.eval(aztime, slantRange);
        aztime += aztimeCor;
    }

    if (slant_range_correction.contains(aztime, slantRange)) {
        const auto srangeCor = slant_range_correction.eval(aztime, slantRange);
        slantRange += srangeCor;
    }

    return flag_converged;
}


/**
* This function fills up a GCOV raster block with NaNs if the block is
# invalid (e.g., outside of the DEM coverage).
*
* @param[in]  block_x            Number of the current block in the X-direction
* @param[in]  block_size_x       Processing block size in the X direction
* @param[in]  block_y            Number of the current block in the Y-direction
* @param[in]  block_size_y       Processing block size in the Y direction
* @param[in]  this_block_size_x  Size of the current block in the X direction
* @param[in]  this_block_size_y  Size of the current block in the Y direction
* @param[out] output_raster      Output raster
*
*/
template<class T>
inline void _fillGcovBlocksWithNans(
    int block_x, int block_size_x, int block_y,
    int block_size_y, int this_block_size_x, int this_block_size_y,
    isce3::io::Raster* output_raster)
{

    // The output raster may be optional (e.g., off-diagonal raster). If
    // it is `nullptr`, return.
    if (output_raster == nullptr) {
        return;
    }

    // declare matrix that will hold the NaNs
    isce3::core::Matrix<T> data_block(this_block_size_y, this_block_size_x);

    // declare variable to hold NaN values according to the templateT,
    // i.e. real (NaN) or complex (NaN, NaN)
    using T_real = typename isce3::real<T>::type;
    T nan_t = 0;
    nan_t *= std::numeric_limits<T_real>::quiet_NaN();

    // fill matrix with NaN
    data_block.fill(nan_t);

    const int nbands = output_raster->numBands();
    for (int band = 0; band < nbands; ++band) {
        _Pragma("omp critical")
        {
            // set block with the matrix `data_block` that
            // is filled with NaNs
            output_raster->setBlock(
                data_block.data(), block_x * block_size_x,
                block_y * block_size_y, this_block_size_x,
                this_block_size_y, band + 1);
        }
    }
}

inline void _saveOptionalFiles(int block_x, int block_size_x, int block_y,
        int block_size_y, int this_block_size_x, int this_block_size_y,
        int block_size_with_upsampling_x, int block_size_with_upsampling_y,
        int this_block_size_with_upsampling_x,
        int this_block_size_with_upsampling_y, isce3::io::Raster* out_geo_rdr,
        isce3::core::Matrix<float>& out_geo_rdr_a,
        isce3::core::Matrix<float>& out_geo_rdr_r,
        isce3::io::Raster* out_geo_dem,
        isce3::core::Matrix<float>& out_geo_dem_array,
        isce3::io::Raster* out_geo_nlooks,
        isce3::core::Matrix<float>& out_geo_nlooks_array,
        isce3::io::Raster* out_geo_rtc,
        isce3::core::Matrix<float>& out_geo_rtc_array,
        isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0,
        isce3::core::Matrix<float>& out_geo_rtc_gamma0_to_sigma0_array,
        isce3::io::Raster* out_mask,
        isce3::core::Matrix<short>& out_mask_array)
{

    if (out_geo_rdr != nullptr)
#pragma omp critical
    {
        out_geo_rdr->setBlock(out_geo_rdr_a.data(),
                block_x * block_size_with_upsampling_x,
                block_y * block_size_with_upsampling_y,
                this_block_size_with_upsampling_x + 1,
                this_block_size_with_upsampling_y + 1, 1);
        out_geo_rdr->setBlock(out_geo_rdr_r.data(),
                block_x * block_size_with_upsampling_x,
                block_y * block_size_with_upsampling_y,
                this_block_size_with_upsampling_x + 1,
                this_block_size_with_upsampling_y + 1, 2);
    }

    if (out_geo_dem != nullptr)
#pragma omp critical
    {
        out_geo_dem->setBlock(out_geo_dem_array.data(),
                block_x * block_size_with_upsampling_x,
                block_y * block_size_with_upsampling_y,
                this_block_size_with_upsampling_x + 1,
                this_block_size_with_upsampling_y + 1, 1);
    }

    if (out_geo_nlooks != nullptr)
#pragma omp critical
    {
        out_geo_nlooks->setBlock(out_geo_nlooks_array.data(),
                block_x * block_size_x, block_y * block_size_y,
                this_block_size_x, this_block_size_y, 1);
    }

    if (out_geo_rtc != nullptr)
#pragma omp critical
    {
        out_geo_rtc->setBlock(out_geo_rtc_array.data(), block_x * block_size_x,
                block_y * block_size_y, this_block_size_x, this_block_size_y,
                1);
    }

    if (out_geo_rtc_gamma0_to_sigma0 != nullptr)
#pragma omp critical
    {
        out_geo_rtc_gamma0_to_sigma0->setBlock(
            out_geo_rtc_gamma0_to_sigma0_array.data(), block_x * block_size_x,
            block_y * block_size_y, this_block_size_x, this_block_size_y,
            1);
    }

    if (out_mask != nullptr)
#pragma omp critical
    {
        out_mask->setBlock(
            out_mask_array.data(),
                block_x * block_size_x, block_y * block_size_y,
                this_block_size_x, this_block_size_y, 1);
    }
}

template<class T>
bool Geocode<T>::_checkLoadEntireRslcCorners(const double y0, const double x0,
        const double yf, const double xf,
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::core::ProjectionBase* proj,
        const std::function<Vec3(double, double,
                const isce3::geometry::DEMInterpolator&,
                isce3::core::ProjectionBase*)>& getDemCoords,
        isce3::geometry::DEMInterpolator& dem_interp, int margin_pixels)
{
    /*
     Check if a geogrid bounding box (y0, x0, yf, xf) fully
     covers the RSLC (represented by the radar_grid).
     */

    const double pixazm = radar_grid.azimuthTimeInterval();
    const double start = radar_grid.sensingStart() - 0.5 * pixazm;
    const double dr = radar_grid.rangePixelSpacing();
    const double r0 = radar_grid.startingRange() - 0.5 * dr;

    double a_min = std::numeric_limits<double>::quiet_NaN();
    double r_min = std::numeric_limits<double>::quiet_NaN();
    double a_max = std::numeric_limits<double>::quiet_NaN();
    double r_max = std::numeric_limits<double>::quiet_NaN();

    std::vector<std::pair<float, float>> vertices_positions = {
            std::make_pair(y0, x0), std::make_pair(y0, xf),
            std::make_pair(yf, x0), std::make_pair(yf, xf)};

    for (auto [dem_y, dem_x] : vertices_positions) {

        double az_time = radar_grid.sensingMid();
        double range_distance = radar_grid.midRange();

        // Convert DEM coordinates (`dem_x` and `dem_y`) from _epsgOut to DEM
        // EPSG coordinates x and y, interpolate height (z), and return:
        // dem_pos_vect = {x, y, z}
        Vec3 dem_pos_vect = getDemCoords(dem_x, dem_y, dem_interp, proj);

        const int converged = isce3::geometry::geo2rdr(
                dem_interp.proj()->inverse(dem_pos_vect), _ellipsoid, _orbit,
                _doppler, az_time, range_distance, radar_grid.wavelength(),
                radar_grid.lookSide(), _threshold, _numiter, 1.0e-8);
        // if it didn't converge, return false
        if (!converged) {
            return false;
        }

        // Convert az. time and range distance to pixel indexes
        double idx_a = (az_time - start) / pixazm;
        double idx_r = (range_distance - r0) / dr;

        /*
        If there is at least one point inside the radar grid,
        do not load entire RSLC
        */
        if (idx_a > margin_pixels &&
                idx_a < radar_grid.length() - 1 - margin_pixels &&
                idx_r > margin_pixels &&
                idx_r < radar_grid.width() - 1 - margin_pixels) {
            return false;
        }

        if (std::isnan(a_min) || idx_a < a_min)
            a_min = idx_a;
        if (std::isnan(a_max) || idx_a > a_max)
            a_max = idx_a;
        if (std::isnan(r_min) || idx_r < r_min)
            r_min = idx_r;
        if (std::isnan(r_max) || idx_r > r_max)
            r_max = idx_r;
    }

    /*
    If no point is inside the RSLC radar grid, we still need to test
    if the bounding box covers the RSLC completely.

    Notice that all points could be located at one side (e.g. East)
    of the radar grid and the previous check would fail to detect
    that the area of interest has no intersection with the RSLC.
    */

    const bool flag_load_entire_rslc =
            (a_min <= margin_pixels && r_min <= margin_pixels &&
                    a_max >= radar_grid.length() - 1 - margin_pixels &&
                    r_max >= radar_grid.width() - 1 - margin_pixels);

    return flag_load_entire_rslc;
}

template<class T>
void Geocode<T>::_getRadarPositionBorder(double geogrid_upsampling,
        const double y0, const double x0, const double yf, const double xf,
        double* a_min, double* r_min, double* a_max, double* r_max,
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::core::ProjectionBase* proj,
        const std::function<Vec3(double, double,
                const isce3::geometry::DEMInterpolator&,
                isce3::core::ProjectionBase*)>& getDemCoords,
        isce3::geometry::DEMInterpolator& dem_interp,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction) {
    /*
    Get radar grid boundaries, i.e. min and max rg. and az. indexes, using
    the border of a geogrid bounding box.
    */

    const int imax = _geoGridLength * geogrid_upsampling;
    const int jmax = _geoGridWidth * geogrid_upsampling;

    double az_time = radar_grid.sensingMid();
    double range_distance = radar_grid.midRange();

    bool flag_direction_line = true, flag_save_vectors = false;
    bool flag_compute_min_max = true;

    _getRadarPositionVect(y0, 0, jmax, geogrid_upsampling, &az_time,
            &range_distance, a_min, r_min, a_max, r_max, radar_grid, proj,
            dem_interp, getDemCoords, flag_direction_line, flag_save_vectors,
            flag_compute_min_max, az_time_correction, slant_range_correction);

    _getRadarPositionVect(yf, 0, jmax, geogrid_upsampling, &az_time,
            &range_distance, a_min, r_min, a_max, r_max, radar_grid, proj,
            dem_interp, getDemCoords, flag_direction_line, flag_save_vectors,
            flag_compute_min_max, az_time_correction, slant_range_correction);

    // pre-compute radar positions on the left side of the geogrid
    flag_direction_line = false;

    int i_start = 1;
    int i_end = imax - 1;

    _getRadarPositionVect(x0, i_start, i_end, geogrid_upsampling, &az_time,
            &range_distance, a_min, r_min, a_max, r_max, radar_grid, proj,
            dem_interp, getDemCoords, flag_direction_line, flag_save_vectors,
            flag_compute_min_max, az_time_correction, slant_range_correction);

    _getRadarPositionVect(xf, i_start, i_end, geogrid_upsampling, &az_time,
            &range_distance, a_min, r_min, a_max, r_max, radar_grid, proj,
            dem_interp, getDemCoords, flag_direction_line, flag_save_vectors,
            flag_compute_min_max, az_time_correction, slant_range_correction);
}

template<class T>
void Geocode<T>::_getRadarGridBoundaries(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& input_raster, isce3::io::Raster& dem_raster,
        isce3::core::ProjectionBase* proj, double geogrid_upsampling,
        bool flag_upsample_radar_grid,
        isce3::core::dataInterpMethod dem_interp_method, int* offset_y,
        int* offset_x, int* grid_size_y, int* grid_size_x)
{
    /*
    Get radar grid boundaries (offsets and window size) based on
    the Geocode object geogrid attributes.
    */

    double y0 = _geoGridStartY;
    double x0 = _geoGridStartX;
    double yf = _geoGridStartY + _geoGridLength * _geoGridSpacingY;
    double xf = _geoGridStartX + _geoGridWidth * _geoGridSpacingX;

    isce3::geometry::DEMInterpolator dem_interp(0, dem_interp_method);

    auto error_code =
            loadDemFromProj(dem_raster, x0, xf, y0, yf, &dem_interp, proj);

    if (error_code != isce3::error::ErrorCode::Success) {
        throw isce3::except::RuntimeError(
                ISCE_SRCINFO(), "ERROR invalid DEM for given area");
    }

    int margin_pixels = 50;

    std::function<Vec3(double, double, const isce3::geometry::DEMInterpolator&,
            isce3::core::ProjectionBase*)>
            getDemCoords;

    if (proj->code() == dem_raster.getEPSG()) {
        getDemCoords = isce3::geometry::getDemCoordsSameEpsg;
    } else {
        getDemCoords = isce3::geometry::getDemCoordsDiffEpsg;
    }

    bool flag_load_entire_rslc = _checkLoadEntireRslcCorners(y0, x0, yf, xf,
            radar_grid, proj, getDemCoords, dem_interp, margin_pixels);

    /*
    If the four courners surround the RSLC, load entire RSLC
    */
    if (flag_load_entire_rslc) {
        *offset_y = 0;
        *offset_x = 0;
        *grid_size_y = radar_grid.length();
        *grid_size_x = radar_grid.width();
        return;
    }

    double a_min = std::numeric_limits<double>::quiet_NaN();
    double r_min = std::numeric_limits<double>::quiet_NaN();
    double a_max = std::numeric_limits<double>::quiet_NaN();
    double r_max = std::numeric_limits<double>::quiet_NaN();

    /*
    Otherwise, use the geogrid bounding box perimeter (borders) to obtain
    the minimum and maximum az. and rg. values
    */
    _getRadarPositionBorder(geogrid_upsampling, y0, x0, yf, xf, &a_min, &r_min,
            &a_max, &r_max, radar_grid, proj, getDemCoords, dem_interp);

    int radar_grid_range_upsampling = flag_upsample_radar_grid ? 2 : 1;

    // azimuth block boundary
    *offset_y = std::min(
            std::max(static_cast<int>(std::floor(a_min) - margin_pixels), 0),
            static_cast<int>(input_raster.length() - 1));
    const int ybound = std::min(
            std::max(static_cast<int>(std::ceil(a_max) + margin_pixels), 0),
            static_cast<int>(input_raster.length() - 1));

    *grid_size_y = ybound - *offset_y + 1;

    // range block boundary
    *offset_x = std::min(
            std::max(static_cast<int>(std::floor(r_min) - margin_pixels), 0),
            static_cast<int>(
                    (input_raster.width() - 1) * radar_grid_range_upsampling));

    const int xbound = std::min(
            std::max(static_cast<int>(std::floor(r_max) + margin_pixels), 0),
            static_cast<int>(
                    (input_raster.width() - 1) * radar_grid_range_upsampling));
    *grid_size_x = xbound - *offset_x + 1;
}

template<class T>
template<class T_out>
void Geocode<T>::geocodeAreaProj(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
        isce3::io::Raster& dem_raster, double geogrid_upsampling,
        bool flag_upsample_radar_grid, bool flag_apply_rtc,
        isce3::geometry::rtcInputTerrainRadiometry input_terrain_radiometry,
        isce3::geometry::rtcOutputTerrainRadiometry output_terrain_radiometry,
        float rtc_min_value_db, double rtc_geogrid_upsampling,
        isce3::geometry::rtcAlgorithm rtc_algorithm,
        isce3::geometry::rtcAreaBetaMode rtc_area_beta_mode,
        double abs_cal_factor,
        float clip_min, float clip_max, float min_nlooks,
        float radar_grid_nlooks, isce3::io::Raster* out_off_diag_terms,
        isce3::io::Raster* out_geo_rdr, isce3::io::Raster* out_geo_dem,
        isce3::io::Raster* out_geo_nlooks, isce3::io::Raster* out_geo_rtc,
        isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        isce3::io::Raster* input_rtc, isce3::io::Raster* output_rtc,
        isce3::io::Raster* input_layover_shadow_mask_raster,
        isce3::product::SubSwaths* sub_swaths,
        isce3::io::Raster* out_mask,
        GeocodeMemoryMode geocode_memory_mode, const long long min_block_size,
        const long long max_block_size,
        isce3::core::dataInterpMethod dem_interp_method)
{

    pyre::journal::info_t info("isce.geocode.GeocodeCov.geocodeAreaProj");
    pyre::journal::error_t error("isce.geocode.GeocodeCov.geocodeAreaProj");

    if (std::isnan(geogrid_upsampling))
        geogrid_upsampling = 1;
    assert(geogrid_upsampling > 0);

    if (flag_upsample_radar_grid && !isce3::is_complex<T>()) {
        std::string error_msg = "radar-grid upsampling is only available";
        error_msg += " for complex inputs";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    if (flag_upsample_radar_grid &&
        std::round(((float) radar_grid.width()) / input_raster.width()) == 1) {
        isce3::product::RadarGridParameters upsampled_radar_grid =
                radar_grid.upsample(1, 2);
        const float upsampled_radar_grid_nlooks = radar_grid_nlooks / 2;
        geocodeAreaProj<T_out>(upsampled_radar_grid, input_raster,
                output_raster, dem_raster, geogrid_upsampling,
                flag_upsample_radar_grid, flag_apply_rtc,
                input_terrain_radiometry, output_terrain_radiometry,
                rtc_min_value_db, rtc_geogrid_upsampling, rtc_algorithm,
                rtc_area_beta_mode,
                abs_cal_factor, clip_min, clip_max, min_nlooks,
                upsampled_radar_grid_nlooks, out_off_diag_terms, out_geo_rdr,
                out_geo_dem, out_geo_nlooks, out_geo_rtc,
                out_geo_rtc_gamma0_to_sigma0,
                az_time_correction, slant_range_correction, input_rtc,
                output_rtc, input_layover_shadow_mask_raster, sub_swaths,
                out_mask, geocode_memory_mode,
                min_block_size, max_block_size, dem_interp_method);
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // number of bands in the input raster
    int nbands = input_raster.numBands();

    int nbands_off_diag_terms = 0;
    if (out_off_diag_terms != nullptr) {
        info << "nbands (diagonal terms): " << nbands << pyre::journal::newline;
        nbands_off_diag_terms = nbands * (nbands - 1) / 2;
        info << "nbands (off-diagonal terms): " << nbands_off_diag_terms
             << pyre::journal::newline;
        assert(out_off_diag_terms->numBands() == nbands_off_diag_terms);
        info << "full covariance: true" << pyre::journal::newline;
        if (!GDALDataTypeIsComplex(input_raster.dtype())){
            std::string error_msg = "Input raster must be complex to"
                                    " generate full-covariance matrix";
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
        }
        if (!GDALDataTypeIsComplex(out_off_diag_terms->dtype())){
            std::string error_msg = "Off-diagonal raster must be complex";
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
        }
    } else {
        info << "nbands: " << nbands << pyre::journal::newline;
        info << "full covariance: false" << pyre::journal::newline;
    }

    if (!std::isnan(clip_min))
        info << "clip min: " << clip_min << pyre::journal::newline;

    if (!std::isnan(clip_max))
        info << "clip max: " << clip_max << pyre::journal::newline;

    if (!std::isnan(min_nlooks))
        info << "nlooks min: " << min_nlooks << pyre::journal::newline;

    // create projection based on epsg code
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(_epsgOut));

    const int imax = _geoGridLength * geogrid_upsampling;
    const int jmax = _geoGridWidth * geogrid_upsampling;

    int offset_y, offset_x, grid_size_y, grid_size_x;

    _getRadarGridBoundaries(radar_grid, input_raster, dem_raster, proj.get(),
            geogrid_upsampling, flag_upsample_radar_grid, dem_interp_method,
            &offset_y, &offset_x, &grid_size_y, &grid_size_x);

    isce3::product::RadarGridParameters radar_grid_cropped =
            radar_grid.offsetAndResize(
                    offset_y, offset_x, grid_size_y, grid_size_x);

    bool is_radar_grid_single_block =
            (geocode_memory_mode !=
                    GeocodeMemoryMode::BlocksGeogridAndRadarGrid);

    // RTC
    isce3::io::Raster* rtc_raster = nullptr;
    isce3::io::Raster* rtc_sigma0_raster = nullptr;
    std::unique_ptr<isce3::io::Raster> rtc_raster_unique_ptr;
    std::unique_ptr<isce3::io::Raster> rtc_raster_sigma0_unique_ptr;

    isce3::core::Matrix<float> rtc_area, rtc_area_sigma;
    if (flag_apply_rtc) {
        std::string input_terrain_radiometry_str =
                get_input_terrain_radiometry_str(input_terrain_radiometry);
        info << "input terrain radiometry: " << input_terrain_radiometry_str
             << pyre::journal::newline;

        std::string output_terrain_radiometry_str =
                get_output_terrain_radiometry_str(output_terrain_radiometry);
        info << "output terrain radiometry: " << output_terrain_radiometry_str
             << pyre::journal::newline;

        if (input_rtc == nullptr) {

            info << "calling RTC (from geocode)..." << pyre::journal::newline;

            // if RTC (area factor) raster does not needed to be saved,
            // initialize it as a GDAL memory virtual file
            if (output_rtc == nullptr) {
                std::string vsimem_ref = (
                    "/vsimem/" + getTempString("geocode_cov_areaproj_rtc"));
                rtc_raster_unique_ptr = std::make_unique<isce3::io::Raster>(
                        vsimem_ref, radar_grid_cropped.width(),
                        radar_grid_cropped.length(), 1, GDT_Float32, "ENVI");
                rtc_raster = rtc_raster_unique_ptr.get();
            }

            // Otherwise, copies the pointer to the output RTC file
            else
                rtc_raster = output_rtc;

            isce3::geometry::rtcAreaMode rtc_area_mode =
                    isce3::geometry::rtcAreaMode::AREA_FACTOR;

            if (std::isnan(rtc_geogrid_upsampling) && flag_upsample_radar_grid)
                rtc_geogrid_upsampling = geogrid_upsampling;
            else if (std::isnan(rtc_geogrid_upsampling))
                rtc_geogrid_upsampling = 2 * geogrid_upsampling;

            isce3::core::MemoryModeBlocksY rtc_memory_mode;
            if (geocode_memory_mode == GeocodeMemoryMode::Auto)
                rtc_memory_mode = isce3::core::MemoryModeBlocksY::AutoBlocksY;
            else if (geocode_memory_mode == GeocodeMemoryMode::SingleBlock)
                rtc_memory_mode = isce3::core::MemoryModeBlocksY::SingleBlockY;
            else
                rtc_memory_mode = isce3::core::MemoryModeBlocksY::MultipleBlocksY;

            if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
                std::string vsimem_ref = (
                    "/vsimem/" + getTempString("geocode_cov_areaproj_rtc_sigma0"));
                rtc_raster_sigma0_unique_ptr = 
                    std::make_unique<isce3::io::Raster>(
                        vsimem_ref, radar_grid.width(),
                        radar_grid.length(), 1, GDT_Float32, "ENVI");
                rtc_sigma0_raster = 
                    rtc_raster_sigma0_unique_ptr.get();
            }

            isce3::io::Raster* out_geo_rdr = nullptr;
            isce3::io::Raster* out_geo_grid = nullptr;

            computeRtc(dem_raster, *rtc_raster, radar_grid_cropped, _orbit,
                    _doppler, _geoGridStartY, _geoGridSpacingY, _geoGridStartX,
                    _geoGridSpacingX, _geoGridLength, _geoGridWidth, _epsgOut,
                    input_terrain_radiometry, output_terrain_radiometry,
                    rtc_area_mode, rtc_algorithm, rtc_area_beta_mode,
                    rtc_geogrid_upsampling, rtc_min_value_db,
                    out_geo_rdr, out_geo_grid,
                    rtc_sigma0_raster, rtc_memory_mode,
                    dem_interp_method, _threshold,
                    _numiter, 1.0e-8, min_block_size, max_block_size);
        } else {
            info << "reading pre-computed RTC..." << pyre::journal::newline;
            rtc_raster = input_rtc;
        }

        if (is_radar_grid_single_block) {
            rtc_area.resize(
                    radar_grid_cropped.length(), radar_grid_cropped.width());
            rtc_raster->getBlock(rtc_area.data(), 0, 0,
                    radar_grid_cropped.width(), radar_grid_cropped.length(), 1);

            if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
                rtc_area_sigma.resize(radar_grid.length(),
                                                radar_grid.width());
                rtc_sigma0_raster->getBlock(
                    rtc_area_sigma.data(), 0, 0, radar_grid.width(),
                    radar_grid.length(), 1);
            }


        }
    }

    isce3::core::Matrix<uint8_t> input_layover_shadow_mask;
    if (input_layover_shadow_mask_raster != nullptr) {
        info << "input layover/shadow mask provided: True" <<
            pyre::journal::newline;

        _validateInputLayoverShadowMaskRaster(
            input_layover_shadow_mask_raster, radar_grid);

        /*
        if we are in radar-grid single block mode, read entire layover/shadow
        mask now. Otherwise, the mask will be read in blocks within
        _runBlock()
        */
        if (is_radar_grid_single_block) {
            input_layover_shadow_mask.resize(
                    radar_grid_cropped.length(), radar_grid_cropped.width());
            input_layover_shadow_mask_raster->getBlock(
                input_layover_shadow_mask.data(), offset_x, offset_y,
                radar_grid_cropped.width(), radar_grid_cropped.length(), 1);
        }
    } else {
        info << "input layover/shadow mask provided: False" <<
            pyre::journal::newline;
    }

    // number of bands in the input raster
    info << "nbands: " << nbands << pyre::journal::newline;

    info << "radar grid width: " << radar_grid_cropped.width()
         << ", length: " << radar_grid_cropped.length()
         << pyre::journal::newline;

    info << "geogrid upsampling: " << geogrid_upsampling << pyre::journal::newline;

    int epsgcode = dem_raster.getEPSG();

    info << "DEM EPSG: " << epsgcode << pyre::journal::endl;
    if (epsgcode < 0) {
        std::string error_msg = "invalid DEM EPSG";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
    info << "output EPSG: " << _epsgOut << pyre::journal::endl;

    info << "reproject DEM (0: false, 1: true): "
         << std::to_string(_epsgOut != dem_raster.getEPSG())
         << pyre::journal::newline;

    info << "apply azimuth offset (0: false, 1: true): "
            << std::to_string(az_time_correction.haveData())
            << pyre::journal::newline;
    info << "apply range offset (0: false, 1: true): "
            << std::to_string(slant_range_correction.haveData())
            << pyre::journal::newline;

    const long long progress_block = ((long long) imax) * jmax / 100;

    double rtc_min_value = 0;

    if (!std::isnan(rtc_min_value_db) && flag_apply_rtc) {
        rtc_min_value = std::pow(10., (rtc_min_value_db / 10.));
        info << "RTC min. value: " << rtc_min_value_db
             << " [dB] = " << rtc_min_value << pyre::journal::newline;
    }

    if (abs_cal_factor != 1)
        info << "absolute calibration factor: " << abs_cal_factor
             << pyre::journal::newline;

    if (radar_grid_nlooks != 1 && out_geo_nlooks != nullptr)
        info << "radar-grid nlooks multiplier: " << radar_grid_nlooks
             << pyre::journal::newline;

    _print_parameters(info, geocode_memory_mode, min_block_size,
                      max_block_size);

    /*
    T - input data template;
    T2 - input template for the _runBlock, that is equal to
         the off-diagonal terms template (if
         applicable) or T2 == T_out (otherwise);
    T_out - diagonal terms template.
    */
    std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>> rdrData;
    std::vector<std::unique_ptr<isce3::core::Matrix<T>>> rdrDataT;

    if (is_radar_grid_single_block) {

        // read slant-range image
        if (std::is_same<T, T_out>::value || nbands_off_diag_terms > 0) {
            _getUpsampledBlock<T, T>(rdrDataT, input_raster, offset_x, offset_y,
                    radar_grid_cropped.width(), radar_grid_cropped.length(),
                    flag_upsample_radar_grid, geocode_memory_mode, min_block_size,
                    max_block_size, info);
        } else {
            _getUpsampledBlock<T, T_out>(rdrData, input_raster, offset_x,
                    offset_y, radar_grid_cropped.width(),
                    radar_grid_cropped.length(), flag_upsample_radar_grid,
                    geocode_memory_mode, min_block_size, max_block_size, info);
        }
    }
    int block_size_x, nblocks_x, block_size_with_upsampling_x;
    int block_size_y, nblocks_y, block_size_with_upsampling_y;
    if (geocode_memory_mode == GeocodeMemoryMode::SingleBlock) {

        nblocks_x = 1;
        block_size_x = _geoGridWidth;
        block_size_with_upsampling_x = jmax;

        nblocks_y = 1;
        block_size_y = _geoGridLength;
        block_size_with_upsampling_y = imax;
    } else {
        isce3::core::getBlockProcessingParametersXY(
                imax, jmax, nbands + nbands_off_diag_terms, sizeof(T_out),
                &info, &block_size_with_upsampling_y, &nblocks_y, 
                &block_size_with_upsampling_x, &nblocks_x,
                min_block_size, max_block_size, geogrid_upsampling);
        block_size_x = block_size_with_upsampling_x / geogrid_upsampling;
        block_size_y = block_size_with_upsampling_y / geogrid_upsampling;
    }

    long long numdone = 0;

    info << "nblocks X: " << nblocks_x << pyre::journal::newline;
    info << "block size X: " << block_size_x << pyre::journal::newline;
    info << "block size X (with upsampling): " << block_size_with_upsampling_x
         << pyre::journal::newline;

    info << "nblocks Y: " << nblocks_y << pyre::journal::newline;
    info << "block size Y: " << block_size_y << pyre::journal::newline;
    info << "block size Y (with upsampling): " << block_size_with_upsampling_y
         << pyre::journal::newline;

    info << "starting geocoding" << pyre::journal::endl;
    if (!std::is_same<T, T_out>::value && nbands_off_diag_terms == 0) {
        _Pragma("omp parallel for schedule(dynamic)")
        for (int block_y = 0; block_y < nblocks_y; ++block_y) {
            for (int block_x = 0; block_x < nblocks_x; ++block_x) {
                _runBlock<T_out, T_out>(radar_grid_cropped,
                        is_radar_grid_single_block, rdrData, block_size_y,
                        block_size_with_upsampling_y, block_y, block_size_x,
                        block_size_with_upsampling_x, block_x, numdone,
                        progress_block, geogrid_upsampling, nbands,
                        nbands_off_diag_terms, dem_interp_method, dem_raster,
                        out_off_diag_terms, out_geo_rdr, out_geo_dem,
                        out_geo_nlooks, out_geo_rtc,
                        out_geo_rtc_gamma0_to_sigma0,
                        proj.get(), flag_apply_rtc,
                        rtc_raster, rtc_sigma0_raster,
                        az_time_correction, slant_range_correction,
                        input_raster, offset_y, offset_x,
                        output_raster, rtc_area, rtc_area_sigma,
                        rtc_min_value, abs_cal_factor,
                        clip_min, clip_max, min_nlooks, radar_grid_nlooks,
                        flag_upsample_radar_grid, input_layover_shadow_mask_raster,
                        input_layover_shadow_mask, sub_swaths, 
                        out_mask, geocode_memory_mode,
                        min_block_size, max_block_size, info);
            }
        }
    } else {
        _Pragma("omp parallel for schedule(dynamic)")
        for (int block_y = 0; block_y < nblocks_y; ++block_y) {
            for (int block_x = 0; block_x < nblocks_x; ++block_x) {
                _runBlock<T, T_out>(radar_grid_cropped,
                        is_radar_grid_single_block, rdrDataT, block_size_y,
                        block_size_with_upsampling_y, block_y, block_size_x,
                        block_size_with_upsampling_x, block_x, numdone,
                        progress_block, geogrid_upsampling, nbands,
                        nbands_off_diag_terms, dem_interp_method, dem_raster,
                        out_off_diag_terms, out_geo_rdr, out_geo_dem,
                        out_geo_nlooks, out_geo_rtc,
                        out_geo_rtc_gamma0_to_sigma0,
                        proj.get(), flag_apply_rtc,
                        rtc_raster, rtc_sigma0_raster,
                        az_time_correction, slant_range_correction,
                        input_raster, offset_y, offset_x,
                        output_raster, rtc_area, rtc_area_sigma,
                        rtc_min_value, abs_cal_factor,
                        clip_min, clip_max, min_nlooks, radar_grid_nlooks,
                        flag_upsample_radar_grid, input_layover_shadow_mask_raster,
                        input_layover_shadow_mask, sub_swaths,
                        out_mask,
                        geocode_memory_mode, min_block_size, max_block_size,
                        info);
            }
        }
    }
    printf("\rgeocode progress: 100%%\n");

    double geotransform[] = {
            _geoGridStartX,  _geoGridSpacingX, 0, _geoGridStartY, 0,
            _geoGridSpacingY};
    if (_geoGridSpacingY > 0) {
        geotransform[3] = _geoGridStartY + _geoGridLength * _geoGridSpacingY;
        geotransform[5] = -_geoGridSpacingY;
    }

    output_raster.setGeoTransform(geotransform);
    output_raster.setEPSG(_epsgOut);

    if (out_geo_rdr != nullptr) {
        double geotransform_edges[] = {_geoGridStartX - _geoGridSpacingX / 2.0,
                                       _geoGridSpacingX / geogrid_upsampling,
                                       0,
                                       _geoGridStartY - _geoGridSpacingY / 2.0,
                                       0,
                                       _geoGridSpacingY / geogrid_upsampling};
        out_geo_rdr->setGeoTransform(geotransform_edges);
        out_geo_rdr->setEPSG(_epsgOut);
    }

    if (out_geo_dem != nullptr) {
        double geotransform_edges[] = {_geoGridStartX - _geoGridSpacingX / 2.0,
                                       _geoGridSpacingX / geogrid_upsampling,
                                       0,
                                       _geoGridStartY - _geoGridSpacingY / 2.0,
                                       0,
                                       _geoGridSpacingY / geogrid_upsampling};
        out_geo_dem->setGeoTransform(geotransform_edges);
        out_geo_dem->setEPSG(_epsgOut);
    }

    if (out_geo_nlooks != nullptr) {
        out_geo_nlooks->setGeoTransform(geotransform);
        out_geo_nlooks->setEPSG(_epsgOut);
    }

    if (out_geo_rtc != nullptr) {
        out_geo_rtc->setGeoTransform(geotransform);
        out_geo_rtc->setEPSG(_epsgOut);
    }

    if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
        out_geo_rtc_gamma0_to_sigma0->setGeoTransform(geotransform);
        out_geo_rtc_gamma0_to_sigma0->setEPSG(_epsgOut);
    }

    if (out_off_diag_terms != nullptr) {
        out_off_diag_terms->setGeoTransform(geotransform);
        out_off_diag_terms->setEPSG(_epsgOut);
    }
    if (out_mask != nullptr) {
        out_mask->setGeoTransform(geotransform);
        out_mask->setEPSG(_epsgOut);
    }

    auto elapsed_time_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    float elapsed_time = ((float) elapsed_time_milliseconds.count()) / 1e3;
    info << "elapsed time (GEO-AP) [s]: " << elapsed_time << pyre::journal::endl;
}

template<class T>
void Geocode<T>::_getRadarPositionVect(double dem_pos_1, const int k_start,
        const int k_end, double geogrid_upsampling, double* az_time,
        double* range_distance, double* y_min, double* x_min, double* y_max,
        double* x_max, const isce3::product::RadarGridParameters& radar_grid,
        isce3::core::ProjectionBase* proj,
        isce3::geometry::DEMInterpolator& dem_interp_block,
        const std::function<Vec3(double, double,
                const isce3::geometry::DEMInterpolator&,
                isce3::core::ProjectionBase*)>& getDemCoords,
        bool flag_direction_line, bool flag_save_vectors,
        bool flag_compute_min_max,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        std::vector<double>* a_vect, std::vector<double>* r_vect,
        std::vector<Vec3>* dem_vect) {
    /*
    Compute radar positions (az, rg, DEM vect.) for a geogrid vector
    (e.g. geogrid border) in X or Y direction (defined by flag_direction_line).
    If flag_compute_min_max is True, the function also return the min/max
    az. and rg. positions
    */

    double pixazm = 0.0, start = 0.0, dr = 0.0, r0 = 0.0;

    if (flag_compute_min_max) {
        // start (az) and r0 at the outer edge of the first pixel
        pixazm = radar_grid.azimuthTimeInterval();
        start = radar_grid.sensingStart() - 0.5 * pixazm;
        dr = radar_grid.rangePixelSpacing();
        r0 = radar_grid.startingRange() - 0.5 * dr;
    }

    for (int kk = k_start; kk <= k_end; ++kk) {
        const int k = kk - k_start;

        Vec3 dem_pos_vect;
        // Convert DEM coordinates (`dem_x` and `dem_y`) from _epsgOut to DEM
        // EPSG coordinates x and y, interpolate height (z), and return:
        // dem_pos_vect = {x, y, z}
        if (flag_direction_line) {
            // flag_direction_line == true: y fixed, varies x
            const double dem_pos_2 =
                    _geoGridStartX + _geoGridSpacingX * kk / geogrid_upsampling;
            dem_pos_vect =
                    getDemCoords(dem_pos_2, dem_pos_1, dem_interp_block, proj);
        } else {
            // flag_direction_line == false: x fixed, varies y
            const double dem_pos_2 =
                    _geoGridStartY + _geoGridSpacingY * kk / geogrid_upsampling;
            dem_pos_vect =
                    getDemCoords(dem_pos_1, dem_pos_2, dem_interp_block, proj);
        }

        // coarse geo2rdr
        int converged =
                _geo2rdrWrapper(dem_interp_block.proj()->inverse(dem_pos_vect),
                        _ellipsoid, _orbit, _doppler, *az_time, *range_distance,
                        radar_grid.wavelength(), radar_grid.lookSide(),
                        az_time_correction, slant_range_correction,
                        _threshold, _numiter, 1.0e-8, true);

        // if it didn't converge, reset initial solution and continue
        if (!converged) {
            *az_time = radar_grid.sensingMid();
            *range_distance = radar_grid.midRange();
            continue;
        }

        // otherwise, save solution
        if (flag_save_vectors) {
            a_vect->operator[](k) = *az_time;
            r_vect->operator[](k) = *range_distance;
            dem_vect->operator[](k) = dem_pos_vect;
        }

        if (!flag_compute_min_max)
            continue;

        // compute min/max pixel indexes
        double y = (*az_time - start) / pixazm;
        double x = (*range_distance - r0) / dr;

        // update min and max rg. and az. indexes
        if (std::isnan(*y_min) || y < *y_min)
            *y_min = y;
        if (std::isnan(*y_max) || y > *y_max)
            *y_max = y;
        if (std::isnan(*x_min) || x < *x_min)
            *x_min = x;
        if (std::isnan(*x_max) || x > *x_max)
            *x_max = x;
    }
}
template<class T>
template<class T2, class T_out>
void Geocode<T>::_runBlock(
        const isce3::product::RadarGridParameters& radar_grid,
        bool is_radar_grid_single_block,
        std::vector<std::unique_ptr<isce3::core::Matrix<T2>>>& rdrData,
        int block_size_y, int block_size_with_upsampling_y, int block_y,
        int block_size_x, int block_size_with_upsampling_x, int block_x,
        long long& numdone, const long long& progress_block,
        double geogrid_upsampling, int nbands, int nbands_off_diag_terms,
        isce3::core::dataInterpMethod dem_interp_method,
        isce3::io::Raster& dem_raster, isce3::io::Raster* out_off_diag_terms,
        isce3::io::Raster* out_geo_rdr, isce3::io::Raster* out_geo_dem,
        isce3::io::Raster* out_geo_nlooks, isce3::io::Raster* out_geo_rtc,
        isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0, 
        isce3::core::ProjectionBase* proj, bool flag_apply_rtc,
        isce3::io::Raster* rtc_raster,
        isce3::io::Raster* rtc_sigma0_raster,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        isce3::io::Raster& input_raster,
        int raster_offset_y, int raster_offset_x,
        isce3::io::Raster& output_raster, isce3::core::Matrix<float>& rtc_area,
        isce3::core::Matrix<float>& rtc_area_sigma,
        float rtc_min_value, double abs_cal_factor, float clip_min,
        float clip_max, float min_nlooks, float radar_grid_nlooks,
        bool flag_upsample_radar_grid,
        isce3::io::Raster* input_layover_shadow_mask_raster,
        isce3::core::Matrix<uint8_t>& input_layover_shadow_mask,
        isce3::product::SubSwaths * sub_swaths,
        isce3::io::Raster* out_mask,
        GeocodeMemoryMode geocode_memory_mode,
        const long long min_block_size, const long long max_block_size,
        pyre::journal::info_t& info)
{

    using isce3::math::complex_operations::operator*;

    // start (az) and r0 at the outer edge of the first pixel
    const double pixazm = radar_grid.azimuthTimeInterval();
    const double start = radar_grid.sensingStart() - 0.5 * pixazm;
    const double dr = radar_grid.rangePixelSpacing();
    const double r0 = radar_grid.startingRange() - 0.5 * dr;

    // set NaN values according to T_out, i.e. real (NaN) or complex (NaN, NaN)
    using T_out_real = typename isce3::real<T_out>::type;
    T_out nan_t_out = 0;
    nan_t_out *= std::numeric_limits<T_out_real>::quiet_NaN();

    double abs_cal_factor_effective;
    if (!isce3::is_complex<T_out>())
        abs_cal_factor_effective = abs_cal_factor;
    else
        abs_cal_factor_effective = std::sqrt(abs_cal_factor);

    int radar_grid_range_upsampling = flag_upsample_radar_grid ? 2 : 1;

    int this_block_size_y = block_size_y;
    if ((block_y + 1) * block_size_y > _geoGridLength)
        this_block_size_y = _geoGridLength % block_size_y;
    const int this_block_size_with_upsampling_y =
            this_block_size_y * geogrid_upsampling;

    int this_block_size_x = block_size_x;
    if ((block_x + 1) * block_size_x > _geoGridWidth)
        this_block_size_x = _geoGridWidth % block_size_x;
    const int this_block_size_with_upsampling_x =
            this_block_size_x * geogrid_upsampling;

    isce3::core::Matrix<float> out_geo_rdr_a;
    isce3::core::Matrix<float> out_geo_rdr_r;
    if (out_geo_rdr != nullptr) {
        out_geo_rdr_a.resize(this_block_size_with_upsampling_y + 1,
                                  this_block_size_with_upsampling_x + 1);
        out_geo_rdr_r.resize(this_block_size_with_upsampling_y + 1,
                                  this_block_size_with_upsampling_x + 1);
        out_geo_rdr_a.fill(std::numeric_limits<float>::quiet_NaN());
        out_geo_rdr_r.fill(std::numeric_limits<float>::quiet_NaN());
    }

    isce3::core::Matrix<float> out_geo_dem_array;
    if (out_geo_dem != nullptr) {
        out_geo_dem_array.resize(this_block_size_with_upsampling_y + 1,
                                      this_block_size_with_upsampling_x + 1);
        out_geo_dem_array.fill(std::numeric_limits<float>::quiet_NaN());
    }

    isce3::core::Matrix<float> out_geo_nlooks_array;
    if (out_geo_nlooks != nullptr) {
        out_geo_nlooks_array.resize(this_block_size_y, this_block_size_x);
        out_geo_nlooks_array.fill(std::numeric_limits<float>::quiet_NaN());
    }

    isce3::core::Matrix<float> out_geo_rtc_array;
    if (out_geo_rtc != nullptr) {
        out_geo_rtc_array.resize(this_block_size_y, this_block_size_x);
        out_geo_rtc_array.fill(std::numeric_limits<float>::quiet_NaN());
    }

    isce3::core::Matrix<float> out_geo_rtc_gamma0_to_sigma0_array;
    if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
        out_geo_rtc_gamma0_to_sigma0_array.resize(this_block_size_y,
                                                  this_block_size_x);
        out_geo_rtc_gamma0_to_sigma0_array.fill(
            std::numeric_limits<float>::quiet_NaN());
    }

    isce3::core::Matrix<short> out_mask_array;
    if (out_mask != nullptr) {
        out_mask_array.resize(
            this_block_size_y, this_block_size_x);
        out_mask_array.fill(0);
    }

    int ii_0 = block_y * block_size_with_upsampling_y;
    int jj_0 = block_x * block_size_with_upsampling_x;

    isce3::geometry::DEMInterpolator dem_interp_block(0, dem_interp_method);

    double minX =
            _geoGridStartX +
            (static_cast<double>(jj_0) / geogrid_upsampling * _geoGridSpacingX);
    double maxX = _geoGridStartX +
                  std::min(static_cast<double>(jj_0) / geogrid_upsampling +
                                   this_block_size_x,
                          static_cast<double>(_geoGridWidth)) *
                          _geoGridSpacingX;

    double minY =
            _geoGridStartY +
            (static_cast<double>(ii_0) / geogrid_upsampling * _geoGridSpacingY);
    double maxY = _geoGridStartY +
                  std::min(static_cast<double>(ii_0) / geogrid_upsampling +
                                   this_block_size_y,
                          static_cast<double>(_geoGridLength)) *
                          _geoGridSpacingY;

    std::function<Vec3(double, double, const isce3::geometry::DEMInterpolator&,
            isce3::core::ProjectionBase*)>
            getDemCoords;

    if (_epsgOut == dem_raster.getEPSG()) {
        getDemCoords = isce3::geometry::getDemCoordsSameEpsg;
    } else {
        getDemCoords = isce3::geometry::getDemCoordsDiffEpsg;
    }

    // Load DEM using the block geogrid extents
    auto error_code = loadDemFromProj(dem_raster, minX, maxX, minY, maxY,
            &dem_interp_block, proj);

    if (error_code != isce3::error::ErrorCode::Success) {

        _fillGcovBlocksWithNans<T_out>(block_x, block_size_x, block_y,
            block_size_y, this_block_size_x, this_block_size_y,
            &output_raster);

        _fillGcovBlocksWithNans<T>(block_x, block_size_x, block_y,
            block_size_y, this_block_size_x, this_block_size_y,
            out_off_diag_terms);

        _saveOptionalFiles(block_x, block_size_x, block_y, block_size_y,
                this_block_size_x, this_block_size_y,
                block_size_with_upsampling_x, block_size_with_upsampling_y,
                this_block_size_with_upsampling_x,
                this_block_size_with_upsampling_y, out_geo_rdr, out_geo_rdr_a,
                out_geo_rdr_r, out_geo_dem, out_geo_dem_array, out_geo_nlooks,
                out_geo_nlooks_array, out_geo_rtc, out_geo_rtc_array,
                out_geo_rtc_gamma0_to_sigma0, out_geo_rtc_gamma0_to_sigma0_array,
                out_mask, out_mask_array);

        return;
    }

    /*
    Example:
    this_block_size_with_upsampling_x = 7 (columns)
    geogrid_upsampling = 1
    this_block_size_y = this_block_size_with_upsampling_y = 4 rows

    - r_last: points to the upper vertices of last processed row (it starts with
              the first row) and it has this_block_size_with_upsampling_x+1
   elements:

       j_00  j_start                                    j_end
       r_last[ 0,    1,    2,    3,    4,    5,    6,    7]: 8 elements

    0: i_00    |-----|-----|-----|-----|-----|-----|-----|
               |  1  |  2  |  3  |  4  |  5  |  6  |  7  |
    1: i_start |-----|-----|-----|-----|-----|-----|-----|
               |     |     |     |     |     |     |     |
    2:         |-----|-----|-----|-----|-----|-----|-----|
               |     |     |     |     |     |     |     |
    3: i_end   |-----|-----|-----|-----|-----|-----|-----|
               |     |     |     |     |     |     |     |
    4:         |-----|-----|-----|-----|-----|-----|-----|

       r_bottom[ 0,    1,    2,    3,    4,    5,    6,    7]

                                                 (geogrid)

   - r_left and r_right:
     r_left and r_right are similar to r_last and r_bottom, with number of
     elements (i_end - i_start) equal to the number of row vertices for each
     column = n_rows + 1 (in the example 5) minus 2:
     n_elements = i_end - i_start = (n_rows + 1) - 2 = n_rows - 1

     since we are working inside the block and with upsampling:
     n_elements = this_block_size_with_upsampling_y - 1
    */

    double a11 = radar_grid.sensingMid();
    double r11 = radar_grid.midRange();
    Vec3 dem11;

    // pre-compute radar positions on the top of the geogrid
    bool flag_direction_line = true, flag_save_vectors = true;
    bool flag_compute_min_max = !is_radar_grid_single_block;

    double a_idx_min = std::numeric_limits<double>::quiet_NaN();
    double r_idx_min = std::numeric_limits<double>::quiet_NaN();
    double a_idx_max = std::numeric_limits<double>::quiet_NaN();
    double r_idx_max = std::numeric_limits<double>::quiet_NaN();

    double dem_y1 =
            _geoGridStartY + _geoGridSpacingY * ii_0 / geogrid_upsampling;
    std::vector<double> a_last(this_block_size_with_upsampling_x + 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_last(this_block_size_with_upsampling_x + 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_last(this_block_size_with_upsampling_x + 1,
                               {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()});
    _getRadarPositionVect(dem_y1, jj_0,
            jj_0 + this_block_size_with_upsampling_x, geogrid_upsampling, &a11,
            &r11, &a_idx_min, &r_idx_min, &a_idx_max, &r_idx_max, radar_grid,
            proj, dem_interp_block, getDemCoords, flag_direction_line,
            flag_save_vectors, flag_compute_min_max, az_time_correction,
            slant_range_correction, &a_last, &r_last, &dem_last);

    // pre-compute radar positions on the bottom of the geogrid
    dem_y1 = (_geoGridStartY +
              (_geoGridSpacingY * (ii_0 + this_block_size_with_upsampling_y) /
               geogrid_upsampling));

    std::vector<double> a_bottom(this_block_size_with_upsampling_x + 1,
                                 std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_bottom(this_block_size_with_upsampling_x + 1,
                                 std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_bottom(this_block_size_with_upsampling_x + 1,
                                 {std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()});
    _getRadarPositionVect(dem_y1, jj_0,
            jj_0 + this_block_size_with_upsampling_x, geogrid_upsampling, &a11,
            &r11, &a_idx_min, &r_idx_min, &a_idx_max, &r_idx_max, radar_grid,
            proj, dem_interp_block, getDemCoords, flag_direction_line,
            flag_save_vectors, flag_compute_min_max, az_time_correction,
            slant_range_correction, &a_bottom, &r_bottom, &dem_bottom);

    // pre-compute radar positions on the left side of the geogrid
    flag_direction_line = false;
    std::vector<double> a_left(this_block_size_with_upsampling_y - 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_left(this_block_size_with_upsampling_y - 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_left(this_block_size_with_upsampling_y - 1,
                               {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()});

    int i_start = (ii_0 + 1);
    int i_end = ii_0 + this_block_size_with_upsampling_y - 1;

    double dem_x1 =
            _geoGridStartX + _geoGridSpacingX * jj_0 / geogrid_upsampling;

    _getRadarPositionVect(dem_x1, i_start, i_end, geogrid_upsampling, &a11,
            &r11, &a_idx_min, &r_idx_min, &a_idx_max, &r_idx_max, radar_grid,
            proj, dem_interp_block, getDemCoords, flag_direction_line,
            flag_save_vectors, flag_compute_min_max, az_time_correction,
            slant_range_correction, &a_left, &r_left, &dem_left);

    // pre-compute radar positions on the right side of the geogrid
    std::vector<double> a_right(this_block_size_with_upsampling_y - 1,
                                std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_right(this_block_size_with_upsampling_y - 1,
                                std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_right(this_block_size_with_upsampling_y - 1,
                                {std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN()});

    dem_x1 = (_geoGridStartX +
              (_geoGridSpacingX * (jj_0 + this_block_size_with_upsampling_x) /
               geogrid_upsampling));

    _getRadarPositionVect(dem_x1, i_start, i_end, geogrid_upsampling, &a11,
            &r11, &a_idx_min, &r_idx_min, &a_idx_max, &r_idx_max, radar_grid,
            proj, dem_interp_block, getDemCoords, flag_direction_line,
            flag_save_vectors, flag_compute_min_max, az_time_correction,
            slant_range_correction, &a_right, &r_right, &dem_right);

    // load radar grid data
    int offset_x = 0, offset_y = 0;
    int xbound = radar_grid.width() - 1;
    int ybound = radar_grid.length() - 1;

    isce3::core::Matrix<float> rtc_area_block, rtc_area_sigma_block;
    isce3::core::Matrix<uint8_t> input_layover_shadow_mask_block;
    std::vector<std::unique_ptr<isce3::core::Matrix<T2>>> rdrDataBlock;
    if (!is_radar_grid_single_block) {

        int margin_pixels = 25;

        // azimuth block boundary
        offset_y = std::min(std::max(static_cast<int>(std::floor(a_idx_min) -
                                                      margin_pixels),
                                    0),
                static_cast<int>(input_raster.length() - 1));
        ybound = std::min(
                std::max(static_cast<int>(std::ceil(a_idx_max) + margin_pixels),
                        0),
                static_cast<int>(input_raster.length() - 1));

        int grid_size_y = ybound - offset_y + 1;

        // range block boundary
        offset_x = std::min(std::max(static_cast<int>(std::floor(r_idx_min) -
                                                      margin_pixels),
                                    0),
                static_cast<int>((input_raster.width() - 1) *
                                 radar_grid_range_upsampling));

        xbound = std::min(std::max(static_cast<int>(std::floor(r_idx_max) +
                                                    margin_pixels),
                                  0),
                static_cast<int>((input_raster.width() - 1) *
                                 radar_grid_range_upsampling));

        int grid_size_x = xbound - offset_x + 1;

        if (grid_size_y <= 0 || grid_size_x <= 0) {

            _fillGcovBlocksWithNans<T_out>(block_x, block_size_x, block_y,
                block_size_y, this_block_size_x, this_block_size_y,
                &output_raster);

            _fillGcovBlocksWithNans<T>(block_x, block_size_x, block_y,
                block_size_y, this_block_size_x, this_block_size_y,
                out_off_diag_terms);

            _saveOptionalFiles(block_x, block_size_x, block_y, block_size_y,
                    this_block_size_x, this_block_size_y,
                    block_size_with_upsampling_x, block_size_with_upsampling_y,
                    this_block_size_with_upsampling_x,
                    this_block_size_with_upsampling_y, out_geo_rdr,
                    out_geo_rdr_a, out_geo_rdr_r, out_geo_dem,
                    out_geo_dem_array, out_geo_nlooks, out_geo_nlooks_array,
                    out_geo_rtc, out_geo_rtc_array,
                    out_geo_rtc_gamma0_to_sigma0, out_geo_rtc_gamma0_to_sigma0_array,
                    out_mask, out_mask_array);

            return;
        }

        isce3::product::RadarGridParameters radar_grid_block =
                radar_grid.offsetAndResize(offset_y, offset_x, grid_size_y,
                                           grid_size_x);

        if (flag_apply_rtc) {
            rtc_area_block.resize(
                    radar_grid_block.length(), radar_grid_block.width());
            rtc_raster->getBlock(rtc_area_block.data(), offset_x, offset_y,
                    radar_grid_block.width(), radar_grid_block.length(), 1);

           if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
                rtc_area_sigma_block.resize(radar_grid.length(),
                                            radar_grid.width());
                rtc_sigma0_raster->getBlock(
                    rtc_area_sigma_block.data(), 0, 0,
                    radar_grid.width(), radar_grid.length(), 1);
            }

        }

        if (input_layover_shadow_mask_raster != nullptr) {
            input_layover_shadow_mask_block.resize(
                    radar_grid_block.length(), radar_grid_block.width());
            input_layover_shadow_mask_raster->getBlock(
                input_layover_shadow_mask_block.data(),
                offset_x, offset_y, radar_grid_block.width(),
                radar_grid_block.length(), 1);
        }

        _getUpsampledBlock<T, T2>(rdrDataBlock, input_raster,
                offset_x + raster_offset_x, offset_y + raster_offset_y,
                radar_grid_block.width(), radar_grid_block.length(),
                flag_upsample_radar_grid, geocode_memory_mode,
                min_block_size, max_block_size, info);
    }

    std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>> geoDataBlock;
    geoDataBlock.reserve(nbands);
    for (int band = 0; band < nbands; ++band)
        geoDataBlock.emplace_back(std::make_unique<isce3::core::Matrix<T_out>>(
                this_block_size_y, this_block_size_x));

    nan_t_out *= std::numeric_limits<T_out_real>::quiet_NaN();
 
    for (int band = 0; band < nbands; ++band)
        geoDataBlock[band]->fill(nan_t_out);

    std::vector<std::unique_ptr<isce3::core::Matrix<T>>> geoDataBlockOffDiag;
    if (nbands_off_diag_terms > 0) {
        geoDataBlockOffDiag.reserve(nbands_off_diag_terms);

        // set NaN values according to T_out, i.e. real (NaN) or complex (NaN,
        // NaN)
        using T_out_real = typename isce3::real<T>::type;
        T nan_t = 0;
        nan_t *= std::numeric_limits<T_out_real>::quiet_NaN();

        for (int band = 0; band < nbands_off_diag_terms; ++band)
            geoDataBlockOffDiag.emplace_back(
                    std::make_unique<isce3::core::Matrix<T>>(
                            this_block_size_y, this_block_size_x));

        for (int band = 0; band < nbands_off_diag_terms; ++band)
            geoDataBlockOffDiag[band]->fill(nan_t);
    }
    /*

         r_last[j], a_last[j]                   r_last[j+1], a_last[j+1]
       -----------|----------------------------------------|
         r01, a01 | r00, a00                      r01, a01 |
                  |                                        |
                  |                                        |
                  |                                        |
                  |                                        |
                  |                 (i, j)                 |
                  |                                        |
                  |                                        |
                  |                                        |
                  |                                        |
                  |                                        |
         r11, a11 | r10, a10                      r11, a11 |
       -----------|----------------------------------------|

       Notice that only the r11 and a11 position that need to be calculated.
       As execution moves to the right. The new r10 and a10 will update their
       values from the previous r11, a11 and so on. The values of the upper
       vertices are obtained from the r_last and a_last vectors.

    */

    for (int i = 0; i < this_block_size_with_upsampling_y; ++i) {

        // initiating lower right vertex
        const int ii = block_y * block_size_with_upsampling_y + i;

        if (i < this_block_size_with_upsampling_y - 1) {
            a11 = a_left[i];
            r11 = r_left[i];
            dem11 = dem_left[i];
        } else {
            a11 = a_bottom[0];
            r11 = r_bottom[0];
            dem11 = dem_bottom[0];
        }

        // initiating lower edge geogrid lat/northing position
        dem_y1 = _geoGridStartY +
                 _geoGridSpacingY * (1.0 + ii) / geogrid_upsampling;

        for (int j = 0; j < this_block_size_with_upsampling_x; ++j) {

            const int jj = block_x * block_size_with_upsampling_x + j;

            _Pragma("omp atomic") numdone++;
            if (numdone % progress_block == 0)
                _Pragma("omp critical")
                {
                    printf("\rgeocode progress: %d%%",
                            static_cast<int>(numdone / progress_block)),
                            fflush(stdout);
                }

            // bottom left (copy from previous bottom right)
            const double a10 = a11;
            const double r10 = r11;
            const Vec3 dem10 = dem11;

            // top left (copy from a_last, r_last, and dem_last)
            const double a00 = a_last[j];
            const double r00 = r_last[j];
            const Vec3 dem00 = dem_last[j];

            // top right (copy from a_last, r_last, and dem_last)
            const double a01 = a_last[j + 1];
            const double r01 = r_last[j + 1];
            const Vec3 dem01 = dem_last[j + 1];

            // update "last" vectors (from lower left vertex)
            a_last[j] = a10;
            r_last[j] = r10;
            dem_last[j] = dem10;

            if (i < this_block_size_with_upsampling_y - 1 &&
                j < this_block_size_with_upsampling_x - 1) {
                // pre-calculate new bottom right
                if (!std::isnan(a10) && !std::isnan(a00) && !std::isnan(a01)) {
                    a11 = a01 + a10 - a00;
                    r11 = r01 + r10 - r00;
                } else if (std::isnan(a11) && !std::isnan(a01)) {
                    a11 = a01;
                    r11 = r01;
                } else if (std::isnan(a11) && !std::isnan(a00)) {
                    a11 = a00;
                    r11 = r00;
                }

                const double dem_x1 =
                        _geoGridStartX +
                        _geoGridSpacingX * (1.0 + jj) / geogrid_upsampling;

                // Convert DEM coordinates (`dem_x` and `dem_y`) from _epsgOut
                // to DEM EPSG coordinates x and y, interpolate height (z), and
                // return: dem11 = {x, y, z}
                dem11 = getDemCoords(dem_x1, dem_y1, dem_interp_block, proj);

                int converged = _geo2rdrWrapper(
                        dem_interp_block.proj()->inverse(dem11), _ellipsoid,
                        _orbit, _doppler, a11, r11, radar_grid.wavelength(),
                        radar_grid.lookSide(), az_time_correction,
                        slant_range_correction, _threshold, _numiter, 1.0e-8);

                if (!converged) {
                    a11 = std::numeric_limits<double>::quiet_NaN();
                    r11 = std::numeric_limits<double>::quiet_NaN();
                }

            } else if (i >= this_block_size_with_upsampling_y - 1 &&
                       !std::isnan(a_bottom[j + 1]) &&
                       !std::isnan(r_bottom[j + 1])) {
                a11 = a_bottom[j + 1];
                r11 = r_bottom[j + 1];
                dem11 = dem_bottom[j + 1];
            } else if (j >= this_block_size_with_upsampling_x - 1 &&
                       !std::isnan(a_right[i]) && !std::isnan(r_right[i])) {
                a11 = a_right[i];
                r11 = r_right[i];
                dem11 = dem_right[i];
            } else {
                a11 = std::numeric_limits<double>::quiet_NaN();
                r11 = std::numeric_limits<double>::quiet_NaN();
            }

            // if last column, also update top-right "last" arrays (from lower
            //   right vertex)
            if (j == this_block_size_with_upsampling_x - 1) {
                a_last[j + 1] = a11;
                r_last[j + 1] = r11;
                dem_last[j + 1] = dem11;
            }

            // save geo-edges
            if (out_geo_dem != nullptr) {
                if (i == 0) {
                    out_geo_dem_array(i, j + 1) = dem01[2];
                }
                if (i == 0 && j == 0) {
                    out_geo_dem_array(i, j) = dem00[2];
                }
                if (j == 0) {
                    out_geo_dem_array(i + 1, j) = dem10[2];
                }
                out_geo_dem_array(i + 1, j + 1) = dem11[2];
            }

            if (std::isnan(a00) || std::isnan(a10) || std::isnan(a10) ||
                    std::isnan(a11)) {
                continue;
            }

            double y00 = (a00 - start) / pixazm;
            double y10 = (a10 - start) / pixazm;
            double y01 = (a01 - start) / pixazm;
            double y11 = (a11 - start) / pixazm;

            double x00 = (r00 - r0) / dr;
            double x10 = (r10 - r0) / dr;
            double x01 = (r01 - r0) / dr;
            double x11 = (r11 - r0) / dr;

            int margin = isce3::core::AREA_PROJECTION_RADAR_GRID_MARGIN;

            // define slant-range window
            const int y_min = std::floor((std::min(std::min(y00, y01),
                                      std::min(y10, y11)))) -
                              1;
            if (y_min < -margin ||
                y_min > ybound + 1)
                continue;
            const int x_min = std::floor((std::min(std::min(x00, x01),
                                      std::min(x10, x11)))) -
                              1;
            if (x_min < -margin ||
                x_min > xbound + 1)
                continue;
            const int y_max = std::ceil((std::max(std::max(y00, y01),
                                      std::max(y10, y11)))) +
                              1;
            if (y_max > ybound + 1 + margin || y_max < -1 || y_max < y_min)
                continue;
            const int x_max = std::ceil((std::max(std::max(x00, x01),
                                      std::max(x10, x11)))) +
                              1;
            if (x_max > xbound + 1 + margin || x_max < -1 || x_max < x_min)
                continue;

            // Crop indexes around (x_min, y_min) and (x_max, y_max)
            // New indexes vary from 0 to (size_x, size_y)
            double y00_cut = y00 - y_min;
            double y10_cut = y10 - y_min;
            double y01_cut = y01 - y_min;
            double y11_cut = y11 - y_min;
            double x00_cut = x00 - x_min;
            double x10_cut = x10 - x_min;
            double x01_cut = x01 - x_min;
            double x11_cut = x11 - x_min;
            const int size_x = x_max - x_min + 1;
            const int size_y = y_max - y_min + 1;

            isce3::core::Matrix<double> w_arr(size_y, size_x);
            w_arr.fill(0);
            double w_total = 0;
            int plane_orientation;
            if (radar_grid.lookSide() == isce3::core::LookSide::Left)
                plane_orientation = -1;
            else
                plane_orientation = 1;

            isce3::geometry::areaProjIntegrateSegment(y00_cut, y01_cut, x00_cut,
                    x01_cut, size_y, size_x, w_arr, w_total, plane_orientation);
            isce3::geometry::areaProjIntegrateSegment(y01_cut, y11_cut, x01_cut,
                    x11_cut, size_y, size_x, w_arr, w_total, plane_orientation);
            isce3::geometry::areaProjIntegrateSegment(y11_cut, y10_cut, x11_cut,
                    x10_cut, size_y, size_x, w_arr, w_total, plane_orientation);
            isce3::geometry::areaProjIntegrateSegment(y10_cut, y00_cut, x10_cut,
                    x00_cut, size_y, size_x, w_arr, w_total, plane_orientation);

            bool flag_self_intersecting_area_element = false;

            // test for self-intersection
            for (int yy = 0; yy < size_y; ++yy) {
                for (int xx = 0; xx < size_x; ++xx) {
                    double w = w_arr(yy, xx);
                    if (w * w_total < 0 && abs(w) >  0.00001) {
                        flag_self_intersecting_area_element = true;
                        break;
                    }
                }
                if (flag_self_intersecting_area_element) {
                    break;
                }
            }

            if (flag_self_intersecting_area_element) {
                /*
                If self-intersecting, divide area element (geogrid pixel) into
                two triangles and integrate them separately.
                */
                isce3::core::Matrix<double> w_arr_1(size_y, size_x);
                w_arr_1.fill(0);
                double w_total_1 = 0;
                isce3::geometry::areaProjIntegrateSegment(y00_cut, y01_cut, x00_cut,
                        x01_cut, size_y, size_x, w_arr_1, w_total_1, plane_orientation);
                isce3::geometry::areaProjIntegrateSegment(y01_cut, y11_cut, x01_cut,
                        x11_cut, size_y, size_x, w_arr_1, w_total_1, plane_orientation);
                isce3::geometry::areaProjIntegrateSegment(y11_cut, y00_cut, x11_cut,
                        x00_cut, size_y, size_x, w_arr_1, w_total_1, plane_orientation);

                isce3::core::Matrix<double> w_arr_2(size_y, size_x);
                w_arr_2.fill(0);
                double w_total_2 = 0;
                isce3::geometry::areaProjIntegrateSegment(y00_cut, y11_cut, x00_cut,
                        x11_cut, size_y, size_x, w_arr_2, w_total_2, plane_orientation);
                isce3::geometry::areaProjIntegrateSegment(y11_cut, y10_cut, x11_cut,
                        x10_cut, size_y, size_x, w_arr_2, w_total_2, plane_orientation);
                isce3::geometry::areaProjIntegrateSegment(y10_cut, y00_cut, x10_cut,
                        x00_cut, size_y, size_x, w_arr_2, w_total_2, plane_orientation);

                w_total = 0;
                /*
                The new weight array `w_arr` is the sum of the absolute values of both
                triangles weighted arrays `w_arr_1` and `w_arr_2`. The integrated
                total `w_total` is updated accordingly.
                */
                for (int yy = 0; yy < size_y; ++yy) {
                    for (int xx = 0; xx < size_x; ++xx) {
                        w_arr(yy, xx) = std::min(
                            abs(w_arr_1(yy, xx)) + abs(w_arr_2(yy, xx)), 1.0);
                        w_total += w_arr(yy, xx);
                    }
                }
            }

            double nlooks = 0;
            float area_total = 0, area_sigma_total = 0;
            std::vector<T_out> cumulative_sum(nbands, 0);
            std::vector<T> cumulative_sum_off_diag_terms(nbands_off_diag_terms,
                                                         0);

            std::vector<int> samples_sub_swath_counts;
            // std::map<short, int> samples_sub_swath_counts;

            // add all slant-range elements that contributes to the geogrid
            // pixel
            for (int yy = 0; yy < size_y; ++yy) {
                for (int xx = 0; xx < size_x; ++xx) {
                    double w = w_arr(yy, xx);
                    int y = yy + y_min;
                    int x = xx + x_min;

                    /* Radar sample does not intersect with projected polygon
                    (geogrid pixel)
                    */
                    if (w == 0)
                        continue;

                    // Radar sample is out of bounds
                    else if (y - offset_y < 0 || x - offset_x < 0 ||
                             y >= ybound || x >= xbound) {
                        continue;
                    }

                    /*
                    If the sample is marked as shadow or layover-and-shadow,
                    we continue to the next position. The input
                    layover/shadow mask, if provided, is available as:
                    - (1) a "single block" if `is_radar_grid_single_block`
                        is true through the variable `input_layover_shadow_mask`
                    - (2) multiple blocks with varying sizes if `is_radar_grid_single_block`
                        is false through the variable `input_layover_shadow_mask_block`

                    The `x` and `y` coordinates locate the point over the radar grid.
                    If the layover shadow block is being read in multiple blocks, we
                    also need to add `offset_x` and `offset_y` that represent the offsets
                    in X- and Y- directions over the radar-grid coordinates.

                    in which case we skip to the next position, i.e., we "break" the 
                    2 inner for-loop bellow (vars: yy and xx) and "continue" from the parent
                    for-loop (var: kk) above.
                    */
                    else if (input_layover_shadow_mask_raster != nullptr &&
                             ((is_radar_grid_single_block &&
                              (input_layover_shadow_mask(y, x) == SHADOW_VALUE ||
                               input_layover_shadow_mask(y, x) == LAYOVER_AND_SHADOW_VALUE)) ||
                              (!is_radar_grid_single_block &&
                              (input_layover_shadow_mask_block(
                                   y - offset_y, x - offset_x) == SHADOW_VALUE ||
                               input_layover_shadow_mask_block(
                                   y - offset_y, x - offset_x) == LAYOVER_AND_SHADOW_VALUE)))) {
                        continue;
                    }

                    short sample_sub_swath = 1;
                    if (sub_swaths != nullptr) {
                        sub_swaths->getSampleSubSwath(y, x);
                    
                        // Check if radar sample is invalid (radar-grid
                        // single-block)
                        if (sub_swaths != nullptr &&
                                sub_swaths->getSampleSubSwath(y, x) == 0) {
                            continue;
                        }
                    }

                    w = std::abs(w);
                    if (flag_apply_rtc) {
                        float rtc_value;
                        if (is_radar_grid_single_block) {
                            rtc_value = rtc_area(y, x);
                        } else {
                            rtc_value =
                                    rtc_area_block(y - offset_y, x - offset_x);
                        }
                        if (std::isnan(rtc_value) || rtc_value < rtc_min_value)
                            continue;

                        nlooks += w;
                        if (isce3::is_complex<T_out>())
                            rtc_value = std::sqrt(rtc_value);
                        area_total += rtc_value * w;

                        if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
                            float rtc_value_sigma;
                            if (is_radar_grid_single_block) {
                                rtc_value_sigma = rtc_area_sigma(y, x);
                            } else {
                                rtc_value_sigma =
                                    rtc_area_sigma_block(y - offset_y, x - offset_x);
                            }
                            area_sigma_total += rtc_value_sigma * w;
                        }
                        w /= rtc_value;
                    } else {
                        nlooks += w;
                    }

                    if (sub_swaths != nullptr && out_mask != nullptr) {
                        for (int s=0; s < (sample_sub_swath -
                                         samples_sub_swath_counts.size()); s++) {
                            samples_sub_swath_counts.push_back(0);
                        }
                        samples_sub_swath_counts[sample_sub_swath - 1]++;
                    }

                    int band_index = 0;
                    for (int band_1 = 0; band_1 < nbands; ++band_1) {
                        T2 v1;
                        if (is_radar_grid_single_block) {
                            v1 = rdrData[band_1]->operator()(
                                    y - offset_y, x - offset_x);
                        } else {
                            v1 = rdrDataBlock[band_1]->operator()(
                                    y - offset_y, x - offset_x);
                        }

                        _accumulate(cumulative_sum[band_1], v1, w);

                        if (nbands_off_diag_terms > 0) {

                            // cov = v1 * conj(v2)
                            for (int band_2 = 0; band_2 < nbands; ++band_2) {
                                if (band_2 <= band_1)
                                    continue;
                                T2 v2;
                                if (is_radar_grid_single_block) {
                                    v2 = rdrData[band_2]->operator()(
                                            y - offset_y, x - offset_x);
                                } else {
                                    v2 = rdrDataBlock[band_2]->operator()(
                                            y - offset_y, x - offset_x);
                                }

                                _accumulate(cumulative_sum_off_diag_terms
                                                    [band_index],
                                            v1 * std::conj(v2), w);
                                band_index++;
                            }
                        }
                    }
                }
                if (std::isnan(nlooks))
                    break;
            }

            // ignoring boundary or low-sampled area elements
            if (std::isnan(nlooks) ||
                nlooks < isce3::core::AREA_PROJECTION_MIN_VALID_SAMPLES_RATIO *
                                 std::abs(w_total) ||
                (!std::isnan(min_nlooks) &&
                 nlooks * radar_grid_nlooks <= min_nlooks))
                continue;

            if (sub_swaths != nullptr && out_mask != nullptr) {
                short max_sub_swath = distance(samples_sub_swath_counts.begin(),
                    max_element(samples_sub_swath_counts.begin(),
                                samples_sub_swath_counts.end()));
                out_mask_array(i, j) = max_sub_swath + 1;
            }

            // save geo-edges
            if (out_geo_rdr != nullptr) {
                // if first (top) line, save top right
                if (i == 0) {
                    // if first (top) line, save top right
                    out_geo_rdr_a(i, j + 1) = y01;
                    out_geo_rdr_r(i, j + 1) = x01;
                }

                // if first (top left) pixel, save top left pixel
                if (i == 0 && j == 0) {
                    // if first (top left) pixel, save top left pixel
                    out_geo_rdr_a(i, j) = y00;
                    out_geo_rdr_r(i, j) = x00;
                }

                // if first (left) column, save lower left
                if (j == 0) {
                    // if first (left) column, save lower left
                    out_geo_rdr_a(i + 1, j) = y10;
                    out_geo_rdr_r(i + 1, j) = x10;
                }

                // save lower left pixel
                out_geo_rdr_a(i + 1, j + 1) = y11;
                out_geo_rdr_r(i + 1, j + 1) = x11;
            }

            // x, y positions are binned by integer quotient (floor)
            const int x = static_cast<int>(j / geogrid_upsampling);
            const int y = static_cast<int>(i / geogrid_upsampling);

            if (flag_apply_rtc) {
                area_total /= nlooks;
                if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
                    area_sigma_total /= nlooks;
                }
            } else {
                area_total = 1;
                area_sigma_total = 1;
            }

            // save nlooks
            if (out_geo_nlooks != nullptr &&
                std::isnan(out_geo_nlooks_array(y, x)))
                out_geo_nlooks_array(y, x) = (radar_grid_nlooks * nlooks);
            else if (out_geo_nlooks != nullptr)
                out_geo_nlooks_array(y, x) += (radar_grid_nlooks * nlooks);

            // save rtc
            if (out_geo_rtc != nullptr && std::isnan(out_geo_rtc_array(y, x)))
                out_geo_rtc_array(y, x) = (area_total / (geogrid_upsampling *
                                                         geogrid_upsampling));
            else if (out_geo_rtc != nullptr)
                out_geo_rtc_array(y, x) += (area_total / (geogrid_upsampling *
                                                          geogrid_upsampling));

            // save rtc (gamma0 to sigma0)
            if (out_geo_rtc_gamma0_to_sigma0 != nullptr) {
                /*
                The RTC area normalization factor (ANF) gamma0 to sigma0
                is computed from the RTC ANF gamma0 to beta0 (or
                sigma0-ellipsoid) `area_total` divided by the RTC ANF sigma0
                to beta0 `area_sigma_total`
                */
                const double rtc_gamma0_to_sigma0 = area_total / area_sigma_total;
                if (std::isnan(out_geo_rtc_gamma0_to_sigma0_array(y, x))) {
                    out_geo_rtc_gamma0_to_sigma0_array(y, x) = (
                        rtc_gamma0_to_sigma0 / (geogrid_upsampling *
                        geogrid_upsampling));
                }
                else {
                    out_geo_rtc_gamma0_to_sigma0_array(y, x) += (
                        rtc_gamma0_to_sigma0 / (geogrid_upsampling *
                        geogrid_upsampling));
                }
            }

            // compute backscatter contribution v and update output arrays

            for (int band = 0; band < nbands; ++band) {
                T_out v = (static_cast<T_out>(
                        (cumulative_sum[band]) * abs_cal_factor_effective /
                        (nlooks * geogrid_upsampling * geogrid_upsampling)));
                if (std::isnan(std::abs(geoDataBlock[band]->operator()(y, x)))) {
                    geoDataBlock[band]->operator()(y, x) = v;
                }
                else {
                    geoDataBlock[band]->operator()(y, x) += v;
                }
            }
            if (nbands_off_diag_terms > 0) {
                for (int band = 0; band < nbands_off_diag_terms; ++band) {
                    T v = (static_cast<T>((cumulative_sum_off_diag_terms[band]) *
                                 abs_cal_factor_effective /
                                 (nlooks * geogrid_upsampling *
                                  geogrid_upsampling)));
                    if (std::isnan(std::abs(
                            geoDataBlockOffDiag[band]->operator()(y, x)))) {
                        geoDataBlockOffDiag[band]->operator()(y, x) = v;
                    }
                    else {
                        geoDataBlockOffDiag[band]->operator()(y, x) += v;
                    }
                }
            }
        }
    }
    for (int band = 0; band < nbands; ++band) {
        for (int i = 0; i < this_block_size_y; ++i) {
            for (int j = 0; j < this_block_size_x; ++j) {
                T_out geo_value = geoDataBlock[band]->operator()(i, j);

                // no data
                if (std::isnan(std::abs(geo_value)))
                    continue;

                // clip min (complex)
                else if (!std::isnan(clip_min) &&
                         std::abs(geo_value) < clip_min &&
                         isce3::is_complex<T_out>())
                    geoDataBlock[band]->operator()(i, j) =
                            (geo_value * clip_min / std::abs(geo_value));

                // clip min (real)
                else if (!std::isnan(clip_min) &&
                         std::abs(geo_value) < clip_min)
                    geoDataBlock[band]->operator()(i, j) = clip_min;

                // clip max (complex)
                else if (!std::isnan(clip_max) &&
                         std::abs(geo_value) > clip_max &&
                         isce3::is_complex<T_out>())
                    geoDataBlock[band]->operator()(i, j) =
                            (geo_value * clip_max / std::abs(geo_value));

                // clip max (real)
                else if (!std::isnan(clip_max) &&
                         std::abs(geo_value) > clip_max)
                    geoDataBlock[band]->operator()(i, j) = clip_max;
            }
        }
        _Pragma("omp critical")
        {
            output_raster.setBlock(geoDataBlock[band]->data(),
                                   block_x * block_size_x,
                                   block_y * block_size_y, this_block_size_x,
                                   this_block_size_y, band + 1);
        }
    }

    geoDataBlock.clear();

    if (nbands_off_diag_terms > 0) {
        for (int band = 0; band < nbands_off_diag_terms; ++band) {
            for (int i = 0; i < this_block_size_y; ++i) {
                for (int j = 0; j < this_block_size_x; ++j) {

                    T geo_value_off_diag =
                            geoDataBlockOffDiag[band]->operator()(i, j);

                    // no data (complex)
                    if (std::isnan(std::abs(geo_value_off_diag)))
                        continue;

                    // clip min (complex)
                    else if (!std::isnan(clip_min) &&
                             std::abs(geo_value_off_diag) < clip_min)
                        geoDataBlockOffDiag[band]->operator()(i, j) =
                                (geo_value_off_diag * clip_min /
                                 std::abs(geo_value_off_diag));

                    // clip max (complex)
                    else if (!std::isnan(clip_max) &&
                             std::abs(geo_value_off_diag) > clip_max)
                        geoDataBlockOffDiag[band]->operator()(i, j) =
                                (geo_value_off_diag * clip_max /
                                 std::abs(geo_value_off_diag));
                }
            }

            _Pragma("omp critical")
            {
                out_off_diag_terms->setBlock(
                        geoDataBlockOffDiag[band]->data(),
                        block_x * block_size_x, block_y * block_size_y,
                        this_block_size_x, this_block_size_y, band + 1);
            }
        }
    }

    geoDataBlockOffDiag.clear();

    _saveOptionalFiles(block_x, block_size_x, block_y, block_size_y,
            this_block_size_x, this_block_size_y, block_size_with_upsampling_x,
            block_size_with_upsampling_y, this_block_size_with_upsampling_x,
            this_block_size_with_upsampling_y, out_geo_rdr, out_geo_rdr_a,
            out_geo_rdr_r, out_geo_dem, out_geo_dem_array, out_geo_nlooks,
            out_geo_nlooks_array, out_geo_rtc, out_geo_rtc_array,
            out_geo_rtc_gamma0_to_sigma0, out_geo_rtc_gamma0_to_sigma0_array,
            out_mask, out_mask_array);
}

/** Convert enum output_mode to string */
std::string _get_geocode_memory_mode_str(
        isce3::core::GeocodeMemoryMode geocode_memory_mode) {
    std::string geocode_memory_mode_str;
    switch (geocode_memory_mode) {
    case isce3::core::GeocodeMemoryMode::SingleBlock:
        geocode_memory_mode_str = "single block";
        break;
    case isce3::core::GeocodeMemoryMode::BlocksGeogrid:
        geocode_memory_mode_str = "blocks geogrid";
        break;
    case isce3::core::GeocodeMemoryMode::BlocksGeogridAndRadarGrid:
        geocode_memory_mode_str = "blocks geogrid and radargrid";
        break;
    case isce3::core::GeocodeMemoryMode::Auto:
        geocode_memory_mode_str = "auto";
        break;
    default:
        std::string error_message = "ERROR invalid geocode memory mode";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_message);
        break;
    }
    return geocode_memory_mode_str;
}

template<class T>
void Geocode<T>::_print_parameters(pyre::journal::info_t& channel, 
                                  isce3::core::GeocodeMemoryMode& geocode_memory_mode,
                                  const long long min_block_size,
                                  const long long max_block_size) {
    channel << "geocode memory mode: "
            << _get_geocode_memory_mode_str(geocode_memory_mode)
            << pyre::journal::newline
            << "min. block size: " << isce3::core::getNbytesStr(min_block_size)
            << pyre::journal::newline
            << "max. block size: " << isce3::core::getNbytesStr(max_block_size)
            << pyre::journal::newline
            << pyre::journal::endl;
}

template class Geocode<float>;
template class Geocode<double>;
template class Geocode<std::complex<float>>;
template class Geocode<std::complex<double>>;


} // namespace geocode
} // namespace isce3
