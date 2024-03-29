//-*- C++ -*-
//-*- coding: utf-8 -*-

#include "RTC.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

#include <isce3/core/Constants.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Projections.h>
#include <isce3/core/TypeTraits.h>
#include <isce3/core/Utilities.h>
#include <isce3/error/ErrorCode.h>
#include <isce3/geocode/GeocodeCov.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/boundingbox.h>
#include <isce3/geometry/geometry.h>
#include <isce3/geometry/loadDem.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/product/RadarGridParameters.h>
#include <isce3/signal/Looks.h>

using isce3::core::cartesian_t;
using isce3::core::Mat3;
using isce3::core::OrbitInterpBorderMode;
using isce3::core::Vec3;

namespace isce3 { namespace geometry {

template<typename T>
void _clip_min_max(T& radar_value, float clip_min, float clip_max)
{

    // no data (complex)
    if (std::abs(radar_value) == 0) {
        radar_value = std::numeric_limits<T>::quiet_NaN();
        return;
    }

    // clip min (real)
    if (!std::isnan(clip_min) && radar_value < clip_min)
        radar_value = clip_min;

    // clip max (real)
    else if (!std::isnan(clip_max) && radar_value > clip_max)
        radar_value = clip_max;
}

template<typename T>
void _clip_min_max(std::complex<T>& radar_value, float clip_min, float clip_max)
{

    /*
    Since std::numeric_limits<T>::quiet_NaN() with
    complex T is (or may be) undefined, we take the "real type"
    if T (i.e. float or double) to create the NaN value and
    multiply it by the current pixel so that the output will be
    real or complex depending on T and will contain NaNs.
    */
    using T_real = typename isce3::real<T>::type;

    // no data (complex)
    if (std::abs(radar_value) == 0) {
        radar_value *= std::numeric_limits<T_real>::quiet_NaN();
        return;
    }

    // clip min (complex)
    if (!std::isnan(clip_min) && std::abs(radar_value) < clip_min)
        // update magnitude without changing the phase
        radar_value *= clip_min / std::abs(radar_value);

    // clip max (complex)
    else if (!std::isnan(clip_max) && std::abs(radar_value) > clip_max)
        // update magnitude without changing the phase
        radar_value *= clip_max / std::abs(radar_value);
}

template<typename T>
void _applyRtc(isce3::io::Raster& input_raster, isce3::io::Raster& input_rtc,
        isce3::io::Raster& output_raster, float rtc_min_value,
        double abs_cal_factor, float clip_min, float clip_max,
        pyre::journal::info_t& info, bool flag_complex_to_real_squared)
{

    int nbands = input_raster.numBands();
    int width = input_raster.width();
    int length = input_raster.length();

    int block_length;
    int nblocks;
    getBlockProcessingParametersY(length, width, nbands, sizeof(T), &info,
                                  &block_length, &nblocks);

    if (std::isnan(rtc_min_value))
        rtc_min_value = 0;

    if (!isnan(abs_cal_factor)) {
        abs_cal_factor = 1.0;
    }

    /*
    Since std::numeric_limits<T>::quiet_NaN() with
    complex T is (or may be) undefined, we take the "real type"
    if T (i.e. float or double) to create the NaN value and
    multiply it by the current pixel so that the output will be
    real or complex depending on T and will contain NaNs.
    */
    using T_real = typename isce3::real<T>::type;

    // for each band in the input:
    for (size_t band = 0; band < nbands; ++band) {
        info << "applying RTC to band: " << band + 1 << "/" << nbands
             << pyre::journal::endl;

        // get a block of data
        _Pragma("omp parallel for schedule(dynamic)")
            for (int block = 0; block < nblocks; ++block) {

            int effective_block_length = block_length;
            if (block * block_length + effective_block_length > length - 1) {
                effective_block_length = length - block * block_length;
            }

            isce3::core::Matrix<float> rtc_ratio(effective_block_length, width);
            _Pragma("omp critical")
            {
                input_rtc.getBlock(rtc_ratio.data(), 0, block * block_length,
                        width, effective_block_length, 1);
            }

            isce3::core::Matrix<T> radar_data_block(block_length, width);
            if (!flag_complex_to_real_squared) {
                _Pragma("omp critical")
                {
                    input_raster.getBlock(radar_data_block.data(), 0,
                            block * block_length, width, effective_block_length,
                            band + 1);
                }
                for (int i = 0; i < effective_block_length; ++i)
                    for (int jj = 0; jj < width; ++jj) {
                        float rtc_ratio_value = rtc_ratio(i, jj);

                        if (std::isnan(rtc_ratio_value) ||
                                rtc_ratio_value < rtc_min_value) {
                            /* assign NaN by multiplication to cast it
                            to the radar_data_block data type. See
                            comments above (outside for loops) */
                            radar_data_block(i, jj) *=
                                    std::numeric_limits<T_real>::quiet_NaN();
                            continue;
                        }

                        float factor = abs_cal_factor / rtc_ratio_value;
                        if (isce3::is_complex<T>())
                            factor = std::sqrt(factor);
                        radar_data_block(i, jj) *= factor;
                        _clip_min_max(
                                radar_data_block(i, jj), clip_min, clip_max);
                    }
            } else {
                isce3::core::Matrix<std::complex<T>> radar_data_block_complex(
                        block_length, width);
                _Pragma("omp critical")
                {
                    input_raster.getBlock(radar_data_block_complex.data(), 0,
                            block * block_length, width, effective_block_length,
                            band + 1);
                }
                for (int i = 0; i < effective_block_length; ++i)
                    for (int jj = 0; jj < width; ++jj) {
                        float rtc_ratio_value = rtc_ratio(i, jj);
                        /* assign NaN by multiplication to cast it
                            to the radar_data_block data type. See
                            comments above (outside for loops) */
                        if (std::isnan(rtc_ratio_value) ||
                                rtc_ratio_value < rtc_min_value) {
                            radar_data_block(i, jj) *=
                                    std::numeric_limits<T_real>::quiet_NaN();
                        }
                        float factor = abs_cal_factor / rtc_ratio_value;
                        std::complex<T> radar_data_complex =
                                radar_data_block_complex(i, jj);
                        radar_data_block(i, jj) =
                                (radar_data_complex.real() *
                                                radar_data_complex.real() +
                                        radar_data_complex.imag() *
                                                radar_data_complex.imag());
                        radar_data_block(i, jj) *= factor;
                        _clip_min_max(
                                radar_data_block(i, jj), clip_min, clip_max);
                    }
            }

            // set output
            _Pragma("omp critical")
            {
                output_raster.setBlock(radar_data_block.data(), 0,
                        block * block_length, width, effective_block_length,
                        band + 1);
            }
        }
    }
}

void _applyRtcMinValueDb(isce3::core::Matrix<float>& out_array,
        float rtc_min_value_db, rtcAreaMode rtc_area_mode,
        pyre::journal::info_t& info)
{
    if (!std::isnan(rtc_min_value_db) &&
            rtc_area_mode == rtcAreaMode::AREA_FACTOR) {
        float rtc_min_value = std::pow(10., (rtc_min_value_db / 10.));
        info << "applying min. RTC value: " << rtc_min_value_db
             << " [dB] ~= " << rtc_min_value << pyre::journal::endl;
        _Pragma("omp parallel for schedule(dynamic) collapse(2)")
            for (int i = 0; i < out_array.length(); ++i)
                for (int j = 0; j < out_array.width(); ++j) {
                    if (out_array(i, j) >= rtc_min_value)
                        continue;
                    _Pragma("omp atomic write") out_array(i, j) =
                        std::numeric_limits<float>::quiet_NaN();
                    }
        info << "... done" << pyre::journal::endl;
    }
}


void _normalizeRtcArea(isce3::core::Matrix<float>& numerator_array,
        const isce3::core::Matrix<float>& denominator_array,
        pyre::journal::info_t& info)
{
    info << "normalizing gamma-naught area..." << pyre::journal::endl;
    _Pragma("omp parallel for schedule(dynamic) collapse(2)") 
        for (int i = 0; i < numerator_array.length(); ++i) 
            for (int j = 0; j < numerator_array.width(); ++j) {
                const float denominator_value = denominator_array(i, j);
                if (denominator_value == 0) {
                    _Pragma("omp atomic write")
                        numerator_array(i, j) =
                            std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
                _Pragma("omp atomic update") 
                    numerator_array(i, j) /= denominator_value;
            }
}

void applyRtc(const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& input_dop,
        isce3::io::Raster& input_raster, isce3::io::Raster& dem_raster,
        isce3::io::Raster& output_raster,
        rtcInputTerrainRadiometry input_terrain_radiometry,
        rtcOutputTerrainRadiometry output_terrain_radiometry, int exponent,
        rtcAreaMode rtc_area_mode, rtcAlgorithm rtc_algorithm,
        rtcAreaBetaMode rtc_area_beta_mode,
        double geogrid_upsampling, float rtc_min_value_db,
        double abs_cal_factor, float clip_min, float clip_max,
        isce3::io::Raster* out_sigma,
        isce3::io::Raster* input_rtc, isce3::io::Raster* output_rtc,
        isce3::core::MemoryModeBlocksY rtc_memory_mode)
{

    if (exponent < 0 || exponent > 2) {
        std::string error_message =
                "ERROR invalid exponent for RTC pre-process. Valid options:";
        error_message += " 0 (auto selection), 1 (linear), or 2 (square).";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_message);
    }

    bool flag_complex_to_real = isce3::signal::verifyComplexToRealCasting(
            input_raster, output_raster, exponent);

    pyre::journal::info_t info("isce.geometry.applyRtc");

    // declare pointer to the raster containing the RTC area normalization
    // factor
    isce3::io::Raster* rtc_raster;
    std::unique_ptr<isce3::io::Raster> rtc_raster_unique_ptr;

    if (input_rtc == nullptr) {

        // if the RTC area normalization factor raster does not needed to be
        // saved, initialize it as a GDAL memory virtual file
        if (output_rtc == nullptr) {
            std::string vsimem_ref = ("/vsimem/" + getTempString("rtc"));

            rtc_raster_unique_ptr = std::make_unique<isce3::io::Raster>(
                    vsimem_ref, radar_grid.width(), radar_grid.length(), 1,
                    GDT_Float32, "ENVI");
            rtc_raster = rtc_raster_unique_ptr.get();
        }

        // Otherwise, copies the pointer to the output RTC file
        else
            rtc_raster = output_rtc;

        info << "calculating RTC..." << pyre::journal::endl;
        computeRtc(radar_grid, orbit, input_dop, dem_raster, *rtc_raster,
                input_terrain_radiometry, output_terrain_radiometry,
                rtc_area_mode, rtc_algorithm, rtc_area_beta_mode,
                geogrid_upsampling, rtc_min_value_db, out_sigma,
                rtc_memory_mode);
    } else {
        info << "reading pre-computed RTC..." << pyre::journal::endl;
        rtc_raster = input_rtc;
    }

    if (abs_cal_factor != 1)
        info << "absolute calibration factor: " << abs_cal_factor
             << pyre::journal::endl;

    float rtc_min_value = 0;
    if (!std::isnan(rtc_min_value_db)) {
        rtc_min_value = std::pow(10., (rtc_min_value_db / 10.));
        info << "applying min. RTC value: " << rtc_min_value_db
             << " [dB] = " << rtc_min_value << pyre::journal::endl;
    }

    if (!std::isnan(clip_min))
        info << "clip min: " << clip_min << pyre::journal::endl;

    if (!std::isnan(clip_max))
        info << "clip max: " << clip_max << pyre::journal::endl;

    if (input_raster.dtype() == GDT_Float32 ||
            (input_raster.dtype() == GDT_CFloat32 && flag_complex_to_real))
        _applyRtc<float>(input_raster, *rtc_raster, output_raster,
                rtc_min_value_db, abs_cal_factor, clip_min, clip_max, info,
                flag_complex_to_real);
    else if (input_raster.dtype() == GDT_Float64 ||
             (input_raster.dtype() == GDT_CFloat64 && flag_complex_to_real))
        _applyRtc<double>(input_raster, *rtc_raster, output_raster,
                rtc_min_value_db, abs_cal_factor, clip_min, clip_max, info,
                flag_complex_to_real);
    else if (input_raster.dtype() == GDT_CFloat32)
        _applyRtc<std::complex<float>>(input_raster, *rtc_raster, output_raster,
                rtc_min_value_db, abs_cal_factor, clip_min, clip_max, info,
                flag_complex_to_real);
    else if (input_raster.dtype() == GDT_CFloat64)
        _applyRtc<std::complex<double>>(input_raster, *rtc_raster,
                output_raster, rtc_min_value_db, abs_cal_factor, clip_min,
                clip_max, info, flag_complex_to_real);
    else {
        std::string error_message =
                "ERROR not implemented for input raster datatype";
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message);
    }
}

double computeUpsamplingFactor(const DEMInterpolator& dem_interp,
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Ellipsoid& ellps)
{

    // Create a projection object from the DEM interpolator
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(dem_interp.epsgCode()));

    // Get middle XY coordinate in DEM coords, lat/lon, and ECEF XYZ
    Vec3 demXY {dem_interp.midX(), dem_interp.midY(), 0.0};

    const Vec3 xyz0 = ellps.lonLatToXyz(proj->inverse(demXY));

    // Repeat for middle coordinate + deltaX
    demXY[0] += dem_interp.deltaX();

    const Vec3 xyz1 = ellps.lonLatToXyz(proj->inverse(demXY));

    // Repeat for middle coordinate + deltaX + deltaY
    demXY[1] += dem_interp.deltaY();

    const Vec3 xyz2 = ellps.lonLatToXyz(proj->inverse(demXY));

    // Estimate width/length of DEM pixel
    const double dx = (xyz1 - xyz0).norm();
    const double dy = (xyz2 - xyz1).norm();

    // Compute upsampling factor (for now, just use spacing in range direction)
    const double upsampling_factor =
            2 * std::max(dx, dy) / radar_grid.rangePixelSpacing();

    return upsampling_factor;
}


void computeRtc(const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& input_dop,
        isce3::io::Raster& dem_raster, isce3::io::Raster& output_raster,
        rtcInputTerrainRadiometry input_terrain_radiometry,
        rtcOutputTerrainRadiometry output_terrain_radiometry,
        rtcAreaMode rtc_area_mode, rtcAlgorithm rtc_algorithm,
        rtcAreaBetaMode rtc_area_beta_mode,
        double geogrid_upsampling, float rtc_min_value_db,
        isce3::io::Raster* out_sigma,
        isce3::core::MemoryModeBlocksY rtc_memory_mode,
        isce3::core::dataInterpMethod interp_method, double threshold,
        int num_iter, double delta_range, const long long min_block_size,
        const long long max_block_size)
{

    double geotransform[6];
    dem_raster.getGeoTransform(geotransform);
    const double dy = geotransform[5];
    const double dx = geotransform[1];

    int epsg = dem_raster.getEPSG();
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(epsg));

    BoundingBox bbox = getGeoBoundingBoxHeightSearch(
            radar_grid, orbit, proj.get(), input_dop);

    const int MARGIN_PIXELS = 20;
    double y0 = bbox.MinY - MARGIN_PIXELS * std::abs(dy);
    double yf = bbox.MaxY + MARGIN_PIXELS * std::abs(dy);
    const double x0 = bbox.MinX - MARGIN_PIXELS * dx;
    const double xf = bbox.MaxX + MARGIN_PIXELS * dx;
    const int geogrid_length = std::abs(std::ceil((yf - y0) / dy));
    const int geogrid_width = std::ceil((xf - x0) / dx);

    if (dy < 0)
        std::swap(y0, yf);

    computeRtc(dem_raster, output_raster, radar_grid, orbit, input_dop, y0, dy,
            x0, dx, geogrid_length, geogrid_width, epsg,
            input_terrain_radiometry, output_terrain_radiometry, rtc_area_mode,
            rtc_algorithm, rtc_area_beta_mode,
            geogrid_upsampling, rtc_min_value_db,
            nullptr, nullptr, out_sigma, rtc_memory_mode,
            interp_method, threshold, num_iter, delta_range, min_block_size,
            max_block_size);
}

void computeRtc(isce3::io::Raster& dem_raster, isce3::io::Raster& output_raster,
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& input_dop, const double y0,
        const double dy, const double x0, const double dx,
        const int geogrid_length, const int geogrid_width, const int epsg,
        rtcInputTerrainRadiometry input_terrain_radiometry,
        rtcOutputTerrainRadiometry output_terrain_radiometry,
        rtcAreaMode rtc_area_mode, rtcAlgorithm rtc_algorithm,
        rtcAreaBetaMode rtc_area_beta_mode,
        double geogrid_upsampling, float rtc_min_value_db,
        isce3::io::Raster* out_geo_rdr,
        isce3::io::Raster* out_geo_grid, isce3::io::Raster* out_sigma,
        isce3::core::MemoryModeBlocksY rtc_memory_mode,
        isce3::core::dataInterpMethod interp_method, double threshold,
        int num_iter, double delta_range, const long long min_block_size,
        const long long max_block_size)
{

    const isce3::product::GeoGridParameters geogrid(
            x0, y0, dx, dy, geogrid_width, geogrid_length, epsg);
    if (rtc_algorithm == rtcAlgorithm::RTC_AREA_PROJECTION) {
        computeRtcAreaProj(dem_raster, output_raster, radar_grid, orbit,
                input_dop, geogrid, input_terrain_radiometry,
                output_terrain_radiometry, rtc_area_mode,
                rtc_area_beta_mode, geogrid_upsampling,
                rtc_min_value_db, out_geo_rdr, out_geo_grid,
                out_sigma, rtc_memory_mode, interp_method, threshold, num_iter,
                delta_range, min_block_size, max_block_size);
    } else if (
        rtc_area_beta_mode == rtcAreaBetaMode::PROJECTION_ANGLE) {
            std::string error_msg = "the area beta mode PROJECTION_ANGLE is not";
            error_msg += "available for RTC with bilinear distribution";
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
        }
    else {
        computeRtcBilinearDistribution(dem_raster, output_raster, radar_grid,
                orbit, input_dop, geogrid, input_terrain_radiometry,
                output_terrain_radiometry, rtc_area_mode,
                geogrid_upsampling, rtc_min_value_db, out_sigma);
    }
}

void areaProjIntegrateSegment(double y1, double y2, double x1, double x2,
        int length, int width, isce3::core::Matrix<double>& w_arr,
        double& nlooks, int plane_orientation)
{
    // if line is vertical or out of boundaries, return
    if (x2 == x1 || (x1 < 0 && x2 < 0) ||
            (x1 >= width - 1 && x2 >= width - 1) || (y1 < 0 && y2 < 0) ||
            (y1 >= length - 1 && y2 >= length - 1) || std::isnan(y1) ||
            std::isnan(y2) || std::isnan(x1) || std::isnan(x2))
        return;

    double slope = (y2 - y1) / (x2 - x1);
    double offset = y1 - slope * x1;
    double x_start, x_end;
    int segment_multiplier;

    // define segment_multiplier of the integration
    if (x2 - x1 > 0) {
        x_start = x1;
        x_end = x2;
        segment_multiplier = plane_orientation;
    } else {
        x_start = x2;
        x_end = x1;
        segment_multiplier = -plane_orientation;
    }

    if (x_start < 0)
        x_start = 0;

    const double x_increment_margin = 0.000001;

    while (x_start < x_end) {

        const double y_start = slope * x_start + offset;
        const double y_start_next =
                slope * (x_start + x_increment_margin) + offset;
        const int x_index = std::floor(x_start);
        int y_index = std::floor(y_start_next);

        if (y_index < 0) {
            x_start = -offset / slope;
            continue;
        }

        if (y_index > length - 1) {
            x_start = (length - 1 - offset) / slope;
            continue;
        }

        // set the integration end point
        double x_next;
        if (slope == 0)
            x_next = x_index + 1;
        else if (slope / std::abs(slope) > 0)
            x_next = std::min(
                    (y_index + 1 - offset) / slope, (double) x_index + 1);
        else
            x_next = std::min((y_index - offset) / slope, (double) x_index + 1);
        x_next = std::min(x_next, x_end);

        if (x_start == x_next || x_index > width - 1)
            break;

        const double y_next = slope * x_next + offset;

        // calculate area (trapezoid) to be added to current pixel
        const double y_center = (y_next + y_start - 2 * y_index) / 2;
        const double area = segment_multiplier * (x_next - x_start) * y_center;

        w_arr(y_index, x_index) += area;
        nlooks += area;

        // add area below current pixel
        while (y_index - 1 >= 0) {
            y_index -= 1;
            const double area = segment_multiplier * (x_next - x_start);
            w_arr(y_index, x_index) += area;
            nlooks += area;
        }
        x_start = x_next;
    }
}

void _addArea(double gamma_naught_area, double sigma_naught_area,
        double beta_naught_area, isce3::core::Matrix<float>& out_gamma_array,
        isce3::core::Matrix<float>& out_beta_array,
        isce3::core::Matrix<float>& out_sigma_array,
        int length, int width, int x_min, int y_min, int size_x, int size_y,
        isce3::core::Matrix<double>& w_arr, double nlooks,
        isce3::core::Matrix<double>& w_arr_out, double& nlooks_out,
        double x_center, double x_left, double x_right, double y_center,
        double y_left, double y_right, int plane_orientation)
{
    areaProjIntegrateSegment(y_left, y_right, x_left, x_right, size_y, size_x,
            w_arr, nlooks, plane_orientation);

    nlooks_out = 0;
    areaProjIntegrateSegment(y_center, y_right, x_center, x_right, size_y,
            size_x, w_arr_out, nlooks_out, plane_orientation);

    for (int ii = 0; ii < size_y; ++ii)
        for (int jj = 0; jj < size_x; ++jj) {
            double w = w_arr(ii, jj) - w_arr_out(ii, jj);
            w_arr(ii, jj) = 0;

            if (w == 0)
                continue;

            int y = ii + y_min;
            int x = jj + x_min;

            if (x < 0 || y < 0 || y >= length || x >= width)
                continue;

            w /= nlooks - nlooks_out;
            _Pragma("omp atomic")
                out_gamma_array(y, x) += w * gamma_naught_area;

            if (out_beta_array.data() != nullptr) {
                _Pragma("omp atomic")
                    out_beta_array(y, x) += w * beta_naught_area;
            }

            if (out_sigma_array.data() != nullptr) {
                _Pragma("omp atomic")
                    out_sigma_array(y, x) += w * sigma_naught_area;
            }

        }
}

double computeFacet(Vec3 xyz_center, Vec3 xyz_left, Vec3 xyz_right,
        Vec3 target_to_sensor_xyz, Vec3 image_normal_xyz,
        rtcAreaMode rtc_area_mode, rtcAreaBetaMode rtc_area_beta_mode,
        double p1, double& p3, double divisor,
        rtcOutputTerrainRadiometry output_terrain_radiometry,
        double & sigma_naught_area,
        double & beta_naught_area)
{
    const Vec3 normal_facet = normalPlane(xyz_center, xyz_left, xyz_right);
    double cos_local_inc_facet = normal_facet.dot(target_to_sensor_xyz);

    p3 = (xyz_center - xyz_right).norm();

    // If facet is not illuminated by radar, skip
    if (cos_local_inc_facet <= 0)
        return 0;

    // Side lengths (keep p3 solution for next iteration)
    const double p2 = (xyz_right - xyz_left).norm();

    // Heron's formula to get area of facets in XYZ coordinates
    const double h = 0.5 * (p1 + p2 + p3);

    sigma_naught_area =
            (std::sqrt(h * (h - p1) * (h - p2) * (h - p3)));

    /*
    In the rtcAreaMode::AREA mode, we'll return a reference area (e.g., A_gamma),
    whereas, in the rtcAreaMode::AREA_FACTOR mode, we'll return the ratio between
    reference area by the area beta A_beta (e.g., A_gamma / A_beta), which
    represents the RTC area normalization factor (ANF).

    We compute the A_beta using the projection angle if the `rtc_area_mode`
    is rtcAreaBetaMode::AUTO or rtcAreaBetaMode::PROJECTION_ANGLE, i.e.,
    if `rtc_area_mode` is different than rtcAreaBetaMode::PIXEL_AREA
    */
    if (rtc_area_mode == rtcAreaMode::AREA_FACTOR &&
            rtc_area_beta_mode != rtcAreaBetaMode::PIXEL_AREA) {
        const double cos_psi_facet = normal_facet.dot(image_normal_xyz);
        beta_naught_area = sigma_naught_area * cos_psi_facet;
    }

    if (output_terrain_radiometry ==
            rtcOutputTerrainRadiometry::SIGMA_NAUGHT) {
        return sigma_naught_area / divisor;
    }

    double gamma_naught_area = cos_local_inc_facet * sigma_naught_area;

    return gamma_naught_area / divisor;
}

void computeRtcBilinearDistribution(isce3::io::Raster& dem_raster,
        isce3::io::Raster& output_raster,
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& input_dop,
        const isce3::product::GeoGridParameters& geogrid,
        rtcInputTerrainRadiometry input_terrain_radiometry,
        rtcOutputTerrainRadiometry output_terrain_radiometry,
        rtcAreaMode rtc_area_mode,
        double upsample_factor, float rtc_min_value_db,
        isce3::io::Raster* out_sigma)
{

    pyre::journal::info_t info("isce.geometry.computeRtcBilinearDistribution");
    auto start_time = std::chrono::high_resolution_clock::now();

    assert(geogrid.spacingY() < 0);
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(geogrid.epsg()));
    const isce3::core::Ellipsoid& ellps = proj->ellipsoid();

    geogrid.print();
    rtcAreaBetaMode rtc_area_beta_mode = rtcAreaBetaMode::PIXEL_AREA;
    print_parameters(info, radar_grid, input_terrain_radiometry,
            output_terrain_radiometry, rtc_area_mode, rtc_area_beta_mode,
            upsample_factor, rtc_min_value_db);

    const double yf = geogrid.startY() + geogrid.length() * geogrid.spacingY();
    const double margin_x = std::abs(geogrid.spacingX()) * 20;
    const double margin_y = std::abs(geogrid.spacingY()) * 20;

    DEMInterpolator dem_interp(
            0, isce3::core::dataInterpMethod::BIQUINTIC_METHOD);
    dem_interp.loadDEM(dem_raster, geogrid.startX() - margin_x,
            geogrid.startX() + geogrid.width() * geogrid.spacingX() + margin_x,
            std::min(geogrid.startY(), yf) - margin_y,
            std::max(geogrid.startY(), yf) + margin_y);

    const double start = radar_grid.sensingStart();
    const double pixazm =
            radar_grid.azimuthTimeInterval(); // azimuth difference per pixel

    const double r0 = radar_grid.startingRange();
    const double dr = radar_grid.rangePixelSpacing();

    // Bounds for valid RDC coordinates
    double xbound = radar_grid.width() - 1.0;
    double ybound = radar_grid.length() - 1.0;

    // Output raster
    isce3::core::Matrix<float> out_array(radar_grid.length(), radar_grid.width());
    out_array.fill(0);

    // Output raster sigma
    isce3::core::Matrix<float> out_sigma_array;
    
    /*
    `out_sigma_array` is only updated if
    `out_sigma` is not `nullptr` AND `output_terrain_radiometry`
    is different than `rtcOutputTerrainRadiometry::SIGMA_NAUGHT`
    */
    bool flag_compute_area_sigma_separately = (out_sigma != nullptr and
                                               output_terrain_radiometry !=
                                       rtcOutputTerrainRadiometry::SIGMA_NAUGHT);
    if (flag_compute_area_sigma_separately) {
        out_sigma_array.resize(radar_grid.length(), radar_grid.width());
        out_sigma_array.fill(0);
    }

    // ------------------------------------------------------------------------
    // Main code: decompose DEM into facets, compute RDC coordinates
    // ------------------------------------------------------------------------

    // Enter loop to read in SLC range/azimuth coordinates and compute area
    std::cout << std::endl;

    if (std::isnan(upsample_factor))
        upsample_factor =
                computeUpsamplingFactor(dem_interp, radar_grid, ellps);

    const size_t imax = geogrid.length() * upsample_factor;
    const size_t jmax = geogrid.width() * upsample_factor;

    const long long progress_block = ((long long) imax) * jmax / 100;
    long long numdone = 0;
    auto side = radar_grid.lookSide();

    std::function<Vec3(double, double, const DEMInterpolator&,
            isce3::core::ProjectionBase*)>
            getDemCoords;

    if (geogrid.epsg() == dem_raster.getEPSG()) {
        getDemCoords = getDemCoordsSameEpsg;
    } else {
        getDemCoords = getDemCoordsDiffEpsg;
    }

    // Loop over DEM facets
    _Pragma("omp parallel for schedule(dynamic)")
        for (size_t ii = 0; ii < imax; ++ii)
    {
        double a = radar_grid.sensingMid();
        double r = radar_grid.midRange();

        // The inner loop is not parallelized in order to keep the previous
        // solution from geo2rdr as the initial guess for the next call to
        // geo2rdr.
        for (size_t jj = 0; jj < jmax; ++jj) {
            _Pragma("omp atomic") numdone++;

            if (numdone % progress_block == 0)
                _Pragma("omp critical")
                    printf("\rRTC progress: %d%%",
                        (int) ((numdone * 1e2 / imax) / jmax)),
                        fflush(stdout);
            // Central DEM coordinates of facets
            const double dem_ymid = geogrid.startY() + geogrid.spacingY() *
                                                               (0.5 + ii) /
                                                               upsample_factor;
            const double dem_xmid = geogrid.startX() + geogrid.spacingX() *
                                                               (0.5 + jj) /
                                                               upsample_factor;

            const Vec3 inputDEM =
                    getDemCoords(dem_xmid, dem_ymid, dem_interp, proj.get());

            // Compute facet-central LLH vector
            const Vec3 inputLLH = dem_interp.proj()->inverse(inputDEM);
            // Should incorporate check on return status here
            int converged = geo2rdr(inputLLH, ellps, orbit, input_dop, a, r,
                    radar_grid.wavelength(), side, 1e-8, 100, 1e-8);
            if (!converged)
                continue;

            float azpix = (a - start) / pixazm;
            float ranpix = (r - r0) / dr;

            // Establish bounds for bilinear weighting model
            const int x1 = (int) std::floor(ranpix);
            const int x2 = x1 + 1;
            const int y1 = (int) std::floor(azpix);
            const int y2 = y1 + 1;

            // Check to see if pixel lies in valid RDC range
            if (ranpix < -1 or x2 > xbound + 1 or azpix < -1 or y2 > ybound + 1)
                continue;

            // Current x/y-coords in DEM
            const double dem_y0 = geogrid.startY() +
                                  geogrid.spacingY() * ii / upsample_factor;
            const double dem_y1 = dem_y0 + geogrid.spacingY() / upsample_factor;
            const double dem_x0 = geogrid.startX() +
                                  geogrid.spacingX() * jj / upsample_factor;
            const double dem_x1 = dem_x0 + geogrid.spacingX() / upsample_factor;

            // Set DEM-coordinate corner vectors
            const Vec3 dem00 =
                    getDemCoords(dem_x0, dem_y0, dem_interp, proj.get());
            const Vec3 dem01 =
                    getDemCoords(dem_x0, dem_y1, dem_interp, proj.get());
            const Vec3 dem10 =
                    getDemCoords(dem_x1, dem_y0, dem_interp, proj.get());
            const Vec3 dem11 =
                    getDemCoords(dem_x1, dem_y1, dem_interp, proj.get());

            // Convert to XYZ
            const Vec3 xyz00 =
                    ellps.lonLatToXyz(dem_interp.proj()->inverse(dem00));
            const Vec3 xyz01 =
                    ellps.lonLatToXyz(dem_interp.proj()->inverse(dem01));
            const Vec3 xyz10 =
                    ellps.lonLatToXyz(dem_interp.proj()->inverse(dem10));
            const Vec3 xyz11 =
                    ellps.lonLatToXyz(dem_interp.proj()->inverse(dem11));

            // Compute normal vectors for each facet
            const Vec3 normal_facet_1 = normalPlane(xyz00, xyz01, xyz10);
            const Vec3 normal_facet_2 = normalPlane(xyz01, xyz11, xyz10);

            // Side lengths
            const double p00_01 = (xyz00 - xyz01).norm();
            const double p00_10 = (xyz00 - xyz10).norm();
            const double p10_01 = (xyz10 - xyz01).norm();
            const double p11_01 = (xyz11 - xyz01).norm();
            const double p11_10 = (xyz11 - xyz10).norm();

            // Semi-perimeters
            const float h1 = 0.5 * (p00_01 + p00_10 + p10_01);
            const float h2 = 0.5 * (p11_01 + p11_10 + p10_01);

            // Heron's formula to get area of facets in XYZ coordinates
            const float AP1 = std::sqrt(
                    h1 * (h1 - p00_01) * (h1 - p00_10) * (h1 - p10_01));
            const float AP2 = std::sqrt(
                    h2 * (h2 - p11_01) * (h2 - p11_10) * (h2 - p10_01));

            // Compute look angle from sensor to ground
            const Vec3 xyz_mid = ellps.lonLatToXyz(inputLLH);
            isce3::core::cartesian_t xyz_plat, vel;
            isce3::error::ErrorCode status = orbit.interpolate(
                    &xyz_plat, &vel, a, OrbitInterpBorderMode::FillNaN);
            if (status != isce3::error::ErrorCode::Success)
                continue;

            const Vec3 lookXYZ = (xyz_plat - xyz_mid).normalized();

            // Compute dot product between each facet and look vector
            double cos_inc_facet_1 = -lookXYZ.dot(normal_facet_1);
            double cos_inc_facet_2 = -lookXYZ.dot(normal_facet_2);

            // If facets are not illuminated by radar, skip
            if (cos_inc_facet_1 <= 0. and cos_inc_facet_2 <= 0.)
                continue;

            // Compute projected area
            double area = 0, area_sigma = 0;

            if (cos_inc_facet_1 > 0 &&
                    output_terrain_radiometry ==
                            rtcOutputTerrainRadiometry::SIGMA_NAUGHT)
                area += AP1;
            else if (cos_inc_facet_1 > 0) {
                area += AP1 * cos_inc_facet_1;
                if (flag_compute_area_sigma_separately) {
                    area_sigma += AP1;
                }
            }
            if (cos_inc_facet_2 > 0 &&
                    output_terrain_radiometry ==
                            rtcOutputTerrainRadiometry::SIGMA_NAUGHT)
                area += AP2;
            else if (cos_inc_facet_2 > 0)
                area += AP2 * cos_inc_facet_2;
            if (area == 0)
                continue;

            // Compute fractional weights from indices
            const double Wr = ranpix - x1;
            const double Wa = azpix - y1;
            const double Wrc = 1. - Wr;
            const double Wac = 1. - Wa;

            if (rtc_area_mode == rtcAreaMode::AREA_FACTOR) {
                // cosine law: c^2 = a^2 + b^2 - 2.a.b.cos(AB)
                // cos(AB) = (a^2 + b^2 - c^2) / 2.a.b
                const double slant_range = (xyz_mid - xyz_plat).norm();
                const double radius_target = xyz_mid.norm();
                const double radius_platform = xyz_plat.norm();
                const double cos_alpha = (
                    (radius_target * radius_target +
                     radius_platform * radius_platform -
                     slant_range * slant_range) /
                    (2 * radius_target * radius_platform));

                const double ground_velocity =
                        cos_alpha * radius_target * vel.norm() / radius_platform;
                const double area_beta = radar_grid.rangePixelSpacing() *
                                         ground_velocity / radar_grid.prf();
                area /= area_beta;
                if (flag_compute_area_sigma_separately) {
                    area_sigma /= area_beta;
                }
            }

            // if if (ranpix < -1 or x2 > xbound+1 or azpix < -1 or y2 >
            // ybound+1)
            if (y1 >= 0 && x1 >= 0) {
                _Pragma("omp atomic") out_array(y1, x1) += area * Wrc * Wac;
            }
            if (y1 >= 0 && x2 <= xbound) {
                _Pragma("omp atomic") out_array(y1, x2) += area * Wr * Wac;
            }
            if (y2 <= ybound && x1 >= 0) {
                _Pragma("omp atomic") out_array(y2, x1) += area * Wrc * Wa;
            }
            if (y2 <= ybound && x2 <= xbound) {
                _Pragma("omp atomic") out_array(y2, x2) += area * Wr * Wa;
            }

            if (flag_compute_area_sigma_separately) {
                if (y1 >= 0 && x1 >= 0) {
                    _Pragma("omp atomic")
                        out_sigma_array(y1, x1) += area_sigma * Wrc * Wac;
                }
                if (y1 >= 0 && x2 <= xbound) {
                    _Pragma("omp atomic")
                        out_sigma_array(y1, x2) += area_sigma * Wr * Wac;
                }
                if (y2 <= ybound && x1 >= 0) {
                    _Pragma("omp atomic")
                        out_sigma_array(y2, x1) += area_sigma * Wrc * Wa;
                }
                if (y2 <= ybound && x2 <= xbound) {
                    _Pragma("omp atomic")
                        out_sigma_array(y2, x2) += area_sigma * Wr * Wa;
                }
            }
        }
    }

    printf("\rRTC progress: 100%%");
    std::cout << std::endl;

    float min_hgt, max_hgt, avg_hgt;

    dem_interp.computeMinMaxMeanHeight(min_hgt, max_hgt, avg_hgt);
    DEMInterpolator flat_interp(avg_hgt);

    if (input_terrain_radiometry ==
            rtcInputTerrainRadiometry::SIGMA_NAUGHT_ELLIPSOID) {
        // Compute the flat earth incidence angle correction
        _Pragma("omp parallel for schedule(dynamic) collapse(2)")
        for (size_t i = 0; i < radar_grid.length(); ++i) {
            for (size_t j = 0; j < radar_grid.width(); ++j) {

                isce3::core::cartesian_t xyz_plat, vel;
                double a = start + i * pixazm;
                isce3::error::ErrorCode status = orbit.interpolate(
                        &xyz_plat, &vel, a, OrbitInterpBorderMode::FillNaN);
                if (status != isce3::error::ErrorCode::Success)
                    continue;

                // Slant range for current pixel
                const double slt_range = r0 + j * dr;

                // Get LLH and XYZ coordinates for this azimuth/range
                isce3::core::cartesian_t targetLLH, targetXYZ;
                targetLLH[2] = avg_hgt; // initialize first guess
                rdr2geo(a, slt_range, 0, orbit, ellps, flat_interp, targetLLH,
                        radar_grid.wavelength(), side, 1e-8, 20, 20);

                // Computation of ENU coordinates around ground target
                ellps.lonLatToXyz(targetLLH, targetXYZ);
                const Vec3 satToGround = targetXYZ - xyz_plat;
                const Mat3 xyz2enu = Mat3::xyzToEnu(targetLLH[1], targetLLH[0]);
                const Vec3 enu = xyz2enu.dot(satToGround);

                // Compute incidence angle components
                const double costheta = std::abs(enu[2]) / enu.norm();
                const double sintheta = std::sqrt(1. - costheta * costheta);

                out_array(i, j) *= sintheta;
            }
        }
    }

    _applyRtcMinValueDb(out_array, rtc_min_value_db, rtc_area_mode, info);

    output_raster.setBlock(
            out_array.data(), 0, 0, radar_grid.width(), radar_grid.length());

    if (flag_compute_area_sigma_separately) {
        out_sigma->setBlock(
            out_sigma_array.data(), 0, 0, radar_grid.width(), radar_grid.length());
    } else if (out_sigma != nullptr){
        out_sigma->setBlock(
            out_array.data(), 0, 0, radar_grid.width(), radar_grid.length());
    }

    auto elapsed_time_milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time);
    float elapsed_time = ((float) elapsed_time_milliseconds.count()) / 1e3;
    info << "elapsed time (RTC-BI) [s]: " << elapsed_time
         << pyre::journal::endl;
}

void _RunBlock(const int jmax, const int block_size,
        const int block_size_with_upsampling, const int block,
        long long& numdone, const long long progress_block,
        const double geogrid_upsampling,
        isce3::core::dataInterpMethod interp_method,
        isce3::io::Raster& dem_raster, isce3::io::Raster* out_geo_rdr,
        isce3::io::Raster* out_geo_grid, const double start,
        const double pixazm, const double dr, double r0, int xbound, int ybound,
        const isce3::product::GeoGridParameters& geogrid,
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::LUT2d<double>& dop,
        const isce3::core::Ellipsoid& ellipsoid,
        const isce3::core::Orbit& orbit, double threshold, int num_iter,
        double delta_range, isce3::core::Matrix<float>& out_gamma_array,
        isce3::core::Matrix<float>& out_beta_array,
        isce3::core::Matrix<float>& out_sigma_array,
        isce3::core::ProjectionBase* proj, rtcAreaMode rtc_area_mode,
        rtcAreaBetaMode rtc_area_beta_mode,
        rtcInputTerrainRadiometry input_terrain_radiometry,
        rtcOutputTerrainRadiometry output_terrain_radiometry)
{

    auto side = radar_grid.lookSide();

    int this_block_size = block_size;
    if ((block + 1) * block_size > geogrid.length())
        this_block_size = geogrid.length() % block_size;

    const int this_block_size_with_upsampling =
            this_block_size * geogrid_upsampling;
    int ii_0 = block * block_size_with_upsampling;

    DEMInterpolator dem_interp_block(0, interp_method);

    std::function<Vec3(double, double, const DEMInterpolator&,
            isce3::core::ProjectionBase*)>
            getDemCoords;

    isce3::core::Matrix<float> out_geo_rdr_a;
    isce3::core::Matrix<float> out_geo_rdr_r;
    if (out_geo_rdr != nullptr) {
        out_geo_rdr_a.resize(this_block_size_with_upsampling + 1, jmax + 1);
        out_geo_rdr_r.resize(this_block_size_with_upsampling + 1, jmax + 1);
        out_geo_rdr_a.fill(std::numeric_limits<float>::quiet_NaN());
        out_geo_rdr_r.fill(std::numeric_limits<float>::quiet_NaN());
    }

    isce3::core::Matrix<float> out_geo_grid_a;
    isce3::core::Matrix<float> out_geo_grid_r;

    if (out_geo_grid != nullptr) {
        out_geo_grid_r.resize(this_block_size_with_upsampling, jmax);
        out_geo_grid_a.resize(this_block_size_with_upsampling, jmax);
        out_geo_grid_r.fill(std::numeric_limits<float>::quiet_NaN());
        out_geo_grid_a.fill(std::numeric_limits<float>::quiet_NaN());
    }

    // Convert margin to meters it not LonLat
    const double minX = geogrid.startX();
    const double maxX = geogrid.startX() + geogrid.spacingX() * geogrid.width();
    double minY =
            geogrid.startY() + (geogrid.spacingY() * ii_0) / geogrid_upsampling;
    double maxY =
            geogrid.startY() +
            (geogrid.spacingY() * (ii_0 + this_block_size_with_upsampling)) /
                    geogrid_upsampling;

    if (geogrid.epsg() == dem_raster.getEPSG()) {
        getDemCoords = getDemCoordsSameEpsg;

    } else {
        getDemCoords = getDemCoordsDiffEpsg;
    }

    auto error_code = loadDemFromProj(
            dem_raster, minX, maxX, minY, maxY, &dem_interp_block, proj);

    if (error_code != isce3::error::ErrorCode::Success) {
        return;
    }

    /*
    The algorithm iterates over the bottom-right vertices. An extra line is
    needed at the beggining to setup first line and first column. The
    algorithm iterates over the lines and previous ("bottom") computations
    are saved as "last" line elements such as a_last, r_last, and dem_last.
    */

    double a11 = radar_grid.sensingMid();
    double r11 = radar_grid.midRange();
    Vec3 dem11;

    std::vector<double> a_last(
            jmax + 1, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_last(
            jmax + 1, std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_last(
            jmax + 1, {std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::quiet_NaN()});

    /*
    Compute first line
    */
    double dem_y1 =
            geogrid.startY() + (geogrid.spacingY() * ii_0) / geogrid_upsampling;

    for (int jj = 0; jj <= jmax; ++jj) {
        const double dem_x1 = geogrid.startX() +
                              (geogrid.spacingX() * jj) / geogrid_upsampling;

        dem11 = getDemCoords(dem_x1, dem_y1, dem_interp_block, proj);
        // course
        int converged = geo2rdr(dem_interp_block.proj()->inverse(dem11),
                ellipsoid, orbit, dop, a11, r11, radar_grid.wavelength(), side,
                threshold, num_iter, delta_range);
        if (!converged) {
            a11 = radar_grid.sensingMid();
            r11 = radar_grid.midRange();
            continue;
        }
        /*
           Accurate geo2rdr:
           This is required because initial guesses (a11 and r11)
           are not as good for border elements. This was causing slightly
           different results for these elements when compared to
           the single-block solution.
        */
        geo2rdr(dem_interp_block.proj()->inverse(dem11), ellipsoid, orbit, dop,
                a11, r11, radar_grid.wavelength(), side, threshold, num_iter,
                delta_range);

        a_last[jj] = a11;
        r_last[jj] = r11;
        dem_last[jj] = dem11;
    }

    for (int i = 0; i < this_block_size_with_upsampling; ++i) {

        const int ii = block * block_size_with_upsampling + i;

        // initial solution for geo2rdr
        if (!std::isnan(a_last[0])) {
            a11 = a_last[0];
            r11 = r_last[0];
        }

        // firt pixel on the left
        const double dem_x1_0 = geogrid.startX();
        const double dem_y1 = geogrid.startY() + geogrid.spacingY() *
                                                         (1.0 + ii) /
                                                         geogrid_upsampling;
        dem11 = getDemCoords(dem_x1_0, dem_y1, dem_interp_block, proj);

        int converged = geo2rdr(dem_interp_block.proj()->inverse(dem11),
                ellipsoid, orbit, dop, a11, r11, radar_grid.wavelength(), side,
                threshold, num_iter, delta_range);
        if (!converged) {
            a11 = std::numeric_limits<double>::quiet_NaN();
            r11 = std::numeric_limits<double>::quiet_NaN();
        }

        for (int jj = 0; jj < (int) jmax; ++jj) {

            _Pragma("omp atomic") numdone++;
            if (numdone % progress_block == 0)
                _Pragma("omp critical") printf("\rRTC progress: %d%%",
                        (int) (numdone / progress_block)),
                        fflush(stdout);

            // bottom left (copy from previous bottom right)
            const double a10 = a11;
            const double r10 = r11;
            const Vec3 dem10 = dem11;

            // top left (copy from a_last, r_last, and dem_last)
            const double a00 = a_last[jj];
            const double r00 = r_last[jj];
            const Vec3 dem00 = dem_last[jj];

            // top right (copy from a_last, r_last, and dem_last)
            const double a01 = a_last[jj + 1];
            const double r01 = r_last[jj + 1];
            const Vec3 dem01 = dem_last[jj + 1];

            // update "last" vectors (from lower left vertex)
            a_last[jj] = a10;
            r_last[jj] = r10;
            dem_last[jj] = dem10;

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

            const double dem_x1 = geogrid.startX() + geogrid.spacingX() *
                                                             (1.0 + jj) /
                                                             geogrid_upsampling;

            dem11 = getDemCoords(dem_x1, dem_y1, dem_interp_block, proj);

            int converged = geo2rdr(dem_interp_block.proj()->inverse(dem11),
                    ellipsoid, orbit, dop, a11, r11, radar_grid.wavelength(),
                    side, threshold, num_iter, delta_range);
            if (!converged) {
                a11 = std::numeric_limits<double>::quiet_NaN();
                r11 = std::numeric_limits<double>::quiet_NaN();
            }

            // if last column also update top-right "last" arrays (from lower
            //   right vertex)
            if (jj == jmax - 1) {
                a_last[jj + 1] = a11;
                r_last[jj + 1] = r11;
                dem_last[jj + 1] = dem11;
            }

            if (std::isnan(a00) || std::isnan(a01) || std::isnan(a10) ||
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

            // define slant-range window
            int margin = AREA_PROJECTION_RADAR_GRID_MARGIN;
            const int y_min = std::floor((std::min(std::min(y00, y01),
                                      std::min(y10, y11)))) -
                              1;
            if (y_min < -margin || y_min > ybound + 1)
                continue;
            const int x_min = std::floor((std::min(std::min(x00, x01),
                                      std::min(x10, x11)))) -
                              1;
            if (x_min < -margin || x_min > xbound + 1)
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

            if (out_geo_rdr != nullptr) {

                // if first (top) line, save top right
                if (i == 0) {
                    out_geo_rdr_a(i, jj + 1) = y01;
                    out_geo_rdr_r(i, jj + 1) = x01;
                }

                // if first (top left) pixel, save top left pixel
                if (i == 0 && jj == 0) {
                    out_geo_rdr_a(i, jj) = y00;
                    out_geo_rdr_r(i, jj) = x00;
                }

                // if first (left) column, save lower left
                if (jj == 0) {
                    out_geo_rdr_a((i + 1), jj) = y10;
                    out_geo_rdr_r((i + 1), jj) = x10;
                }

                // save lower left pixel
                out_geo_rdr_a((i + 1), jj + 1) = y11;
                out_geo_rdr_r((i + 1), jj + 1) = x11;
            }

            // calculate center point
            const double dem_y = geogrid.startY() + geogrid.spacingY() *
                                                            (0.5 + ii) /
                                                            geogrid_upsampling;
            const double dem_x = geogrid.startX() + geogrid.spacingX() *
                                                            (0.5 + jj) /
                                                            geogrid_upsampling;
            const Vec3 dem_c =
                    getDemCoords(dem_x, dem_y, dem_interp_block, proj);

            double a_c = (a00 + a01 + a10 + a11) / 4.0;
            double r_c = (r00 + r01 + r10 + r11) / 4.0;

            converged = geo2rdr(dem_interp_block.proj()->inverse(dem_c),
                    ellipsoid, orbit, dop, a_c, r_c, radar_grid.wavelength(),
                    side, threshold, num_iter, delta_range);

            if (!converged) {
                a_c = std::numeric_limits<double>::quiet_NaN();
                r_c = std::numeric_limits<double>::quiet_NaN();
            }
            double y_c = (a_c - start) / pixazm;
            double x_c = (r_c - r0) / dr;

            if (out_geo_grid != nullptr) {
                out_geo_grid_a(i, jj) = y_c;
                out_geo_grid_r(i, jj) = x_c;
            }
            if (!converged)
                continue;

            // Set DEM-coordinate corner vectors
            const Vec3 xyz00 = ellipsoid.lonLatToXyz(
                    dem_interp_block.proj()->inverse(dem00));
            const Vec3 xyz10 = ellipsoid.lonLatToXyz(
                    dem_interp_block.proj()->inverse(dem10));
            const Vec3 xyz01 = ellipsoid.lonLatToXyz(
                    dem_interp_block.proj()->inverse(dem01));
            const Vec3 xyz11 = ellipsoid.lonLatToXyz(
                    dem_interp_block.proj()->inverse(dem11));
            const Vec3 target_llh = dem_interp_block.proj()->inverse(dem_c);
            const Vec3 xyz_c = ellipsoid.lonLatToXyz(target_llh);

            // Calculate look vector
            isce3::core::cartesian_t xyz_plat, vel;
            isce3::error::ErrorCode status = orbit.interpolate(
                    &xyz_plat, &vel, a_c, OrbitInterpBorderMode::FillNaN);
            if (status != isce3::error::ErrorCode::Success)
                continue;

            const Vec3 target_to_sensor_xyz = (xyz_plat - xyz_c).normalized();

            // Prepare call to computeFacet()
            double p00_c = (xyz00 - xyz_c).norm();
            double p10_c, p01_c, p11_c, divisor = 1;

            if (rtc_area_mode == rtcAreaMode::AREA_FACTOR &&
                    rtc_area_beta_mode == rtcAreaBetaMode::PIXEL_AREA) {
                // cosine law: c^2 = a^2 + b^2 - 2.a.b.cos(AB)
                // cos(AB) = (a^2 + b^2 - c^2) / 2.a.b
                const double slant_range = (xyz_c - xyz_plat).norm();
                const double radius_target = xyz_c.norm();
                const double radius_platform = xyz_plat.norm();
                const double cos_alpha = (
                    (radius_target * radius_target +
                     radius_platform * radius_platform -
                     slant_range * slant_range) /
                    (2 * radius_target * radius_platform));

                const double ground_velocity =
                        cos_alpha * radius_target * vel.norm() / radius_platform;
                divisor = (radar_grid.rangePixelSpacing() * ground_velocity *
                           radar_grid.azimuthTimeInterval());
            }

            if (input_terrain_radiometry ==
                    rtcInputTerrainRadiometry::SIGMA_NAUGHT_ELLIPSOID) {

                // Computation in ENU coordinates around target
                const Mat3 xyz2enu =
                        Mat3::xyzToEnu(target_llh[1], target_llh[0]);
                const Vec3 target_to_sensor_enu =
                        xyz2enu.dot(target_to_sensor_xyz);
                const double cos_inc = std::abs(target_to_sensor_enu[2]) /
                                       target_to_sensor_enu.norm();

                // Compute incidence angle components
                const double sin_inc = std::sqrt(1. - cos_inc * cos_inc);
                divisor /= sin_inc;
            }

            // Prepare call to _addArea()
            int size_x = x_max - x_min + 1;
            int size_y = y_max - y_min + 1;
            isce3::core::Matrix<double> w_arr_1(size_y, size_x);
            isce3::core::Matrix<double> w_arr_2(size_y, size_x);
            w_arr_1.fill(0);
            w_arr_2.fill(0);

            double nlooks_1 = 0, nlooks_2 = 0;

            double y00_cut = y00 - y_min;
            double y10_cut = y10 - y_min;
            double y01_cut = y01 - y_min;
            double y11_cut = y11 - y_min;
            double y_c_cut = y_c - y_min;

            double x00_cut = x00 - x_min;
            double x10_cut = x10 - x_min;
            double x01_cut = x01 - x_min;
            double x11_cut = x11 - x_min;
            double x_c_cut = x_c - x_min;

            int plane_orientation;
            if (radar_grid.lookSide() == isce3::core::LookSide::Left)
                plane_orientation = -1;
            else
                plane_orientation = 1;

            const Vec3 image_normal_xyz = (plane_orientation *
                (vel.cross(target_to_sensor_xyz)).normalized());

            areaProjIntegrateSegment(y_c_cut, y00_cut, x_c_cut, x00_cut, size_y,
                    size_x, w_arr_1, nlooks_1, plane_orientation);

            // Compute the area (first facet)
            double sigma_naught_area, beta_naught_area;
            double gamma_naught_area =
                    computeFacet(xyz_c, xyz00, xyz01, target_to_sensor_xyz,
                            image_normal_xyz, rtc_area_mode,
                            rtc_area_beta_mode, p00_c, p01_c, divisor,
                            output_terrain_radiometry, sigma_naught_area,
                            beta_naught_area);

            // Add gamma_naught_area to output grid
            _addArea(gamma_naught_area, sigma_naught_area, beta_naught_area,
                    out_gamma_array, out_beta_array, out_sigma_array,
                    radar_grid.length(), radar_grid.width(), x_min, y_min,
                    size_x, size_y, w_arr_1, nlooks_1, w_arr_2, nlooks_2,
                    x_c_cut, x00_cut, x01_cut, y_c_cut, y00_cut, y01_cut,
                    plane_orientation);

            // Compute the area (second facet)
            gamma_naught_area = computeFacet(xyz_c, xyz01, xyz11,
                    target_to_sensor_xyz, image_normal_xyz, rtc_area_mode,
                    rtc_area_beta_mode,
                    p01_c, p11_c, divisor, output_terrain_radiometry,
                    sigma_naught_area, beta_naught_area);

            // Add area to output grid
            _addArea(gamma_naught_area, sigma_naught_area, beta_naught_area,
                    out_gamma_array, out_beta_array, out_sigma_array,
                    radar_grid.length(), radar_grid.width(), x_min, y_min,
                    size_x, size_y, w_arr_2, nlooks_2, w_arr_1, nlooks_1,
                    x_c_cut, x01_cut, x11_cut, y_c_cut, y01_cut, y11_cut,
                    plane_orientation);

            // Compute the area (third facet)
            gamma_naught_area = computeFacet(xyz_c, xyz11, xyz10,
                    target_to_sensor_xyz, image_normal_xyz, rtc_area_mode,
                    rtc_area_beta_mode,
                    p11_c, p10_c, divisor, output_terrain_radiometry,
                    sigma_naught_area, beta_naught_area);

            // Add area to output grid
            _addArea(gamma_naught_area, sigma_naught_area, beta_naught_area,
                    out_gamma_array, out_beta_array, out_sigma_array,
                    radar_grid.length(), radar_grid.width(), x_min, y_min,
                    size_x, size_y, w_arr_1, nlooks_1, w_arr_2, nlooks_2,
                    x_c_cut, x11_cut, x10_cut, y_c_cut, y11_cut, y10_cut,
                    plane_orientation);

            // Compute the area (fourth facet)
            gamma_naught_area = computeFacet(xyz_c, xyz10, xyz00,
                    target_to_sensor_xyz, image_normal_xyz, rtc_area_mode,
                    rtc_area_beta_mode,
                    p10_c, p00_c, divisor, output_terrain_radiometry,
                    sigma_naught_area, beta_naught_area);

            // Add area to output grid
            _addArea(gamma_naught_area, sigma_naught_area, beta_naught_area,
                    out_gamma_array, out_beta_array, out_sigma_array,
                    radar_grid.length(), radar_grid.width(), x_min, y_min,
                    size_x, size_y, w_arr_2, nlooks_2, w_arr_1, nlooks_1,
                    x_c_cut, x10_cut, x00_cut, y_c_cut, y10_cut, y00_cut,
                    plane_orientation);
        }
    }

    if (out_geo_rdr != nullptr)
        _Pragma("omp critical")
        {
            out_geo_rdr->setBlock(out_geo_rdr_a.data(), 0,
                    block * block_size_with_upsampling, jmax + 1,
                    this_block_size_with_upsampling + 1, 1);
            out_geo_rdr->setBlock(out_geo_rdr_r.data(), 0,
                    block * block_size_with_upsampling, jmax + 1,
                    this_block_size_with_upsampling + 1, 2);
        }

    if (out_geo_grid != nullptr)
        _Pragma("omp critical")
        {
            out_geo_grid->setBlock(out_geo_grid_a.data(), 0,
                    block * block_size_with_upsampling, jmax, this_block_size,
                    1);
            out_geo_grid->setBlock(out_geo_grid_r.data(), 0,
                    block * block_size_with_upsampling, jmax, this_block_size,
                    2);
        }
}

void computeRtcAreaProj(isce3::io::Raster& dem_raster,
        isce3::io::Raster& output_raster,
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& input_dop,
        const isce3::product::GeoGridParameters& geogrid,
        rtcInputTerrainRadiometry input_terrain_radiometry,
        rtcOutputTerrainRadiometry output_terrain_radiometry,
        rtcAreaMode rtc_area_mode, rtcAreaBetaMode rtc_area_beta_mode,
        double geogrid_upsampling, float rtc_min_value_db,
        isce3::io::Raster* out_geo_rdr, isce3::io::Raster* out_geo_grid,
        isce3::io::Raster* out_sigma, isce3::core::MemoryModeBlocksY rtc_memory_mode,
        isce3::core::dataInterpMethod interp_method, double threshold,
        int num_iter, double delta_range, const long long min_block_size,
        const long long max_block_size)
{
    /*
      Description of the area projection algorithm can be found in Geocode.cpp
    */

    pyre::journal::info_t info("isce.geometry.computeRtcAreaProj");

    auto start_time = std::chrono::high_resolution_clock::now();

    if (std::isnan(geogrid_upsampling))
        geogrid_upsampling = 2;

    assert(geogrid.length() > 0);
    assert(geogrid.width() > 0);
    assert(geogrid_upsampling > 0);
    assert(geogrid.spacingY() < 0);

    // Ellipsoid being used for processing
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(geogrid.epsg()));
    const isce3::core::Ellipsoid& ellipsoid = proj->ellipsoid();

    geogrid.print();
    print_parameters(info, radar_grid, input_terrain_radiometry,
            output_terrain_radiometry, rtc_area_mode, rtc_area_beta_mode,
            geogrid_upsampling, rtc_min_value_db);

    int epsgcode = dem_raster.getEPSG();
    info << "DEM EPSG: " << epsgcode << pyre::journal::endl;
    if (epsgcode < 0) {
        std::string error_msg = "invalid DEM EPSG";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
    info << "output EPSG: " << geogrid.epsg() << pyre::journal::endl;
    info << "reproject DEM (0: false, 1: true): "
         << std::to_string(geogrid.epsg() != dem_raster.getEPSG())
         << pyre::journal::newline;

    // start (az) and r0 at the outer edge of the first pixel:
    const double pixazm = radar_grid.azimuthTimeInterval();
    double start = radar_grid.sensingStart() - 0.5 * pixazm;
    const double dr = radar_grid.rangePixelSpacing();
    double r0 = radar_grid.startingRange() - 0.5 * dr;

    // Bounds for valid RDC coordinates
    int xbound = radar_grid.width() - 1.0;
    int ybound = radar_grid.length() - 1.0;

    const int imax = geogrid.length() * geogrid_upsampling;
    const int jmax = geogrid.width() * geogrid_upsampling;

    // Output raster
    using T = float;
    isce3::core::Matrix<T> out_gamma_array(radar_grid.length(), radar_grid.width());
    out_gamma_array.fill(0);

    isce3::core::Matrix<T> out_beta_array;
    if (rtc_area_mode == rtcAreaMode::AREA_FACTOR &&
            rtc_area_beta_mode != rtcAreaBetaMode::PIXEL_AREA) {
        out_beta_array.resize(radar_grid.length(), radar_grid.width());
        out_beta_array.fill(0);
    }

    isce3::core::Matrix<float> out_sigma_array;
    if (out_sigma != nullptr) {
        out_sigma_array.resize(radar_grid.length(), radar_grid.width());
        out_sigma_array.fill(0);
    }

    const long long progress_block = ((long long) imax) * jmax / 100;
    long long numdone = 0;
    int block_length, block_length_with_upsampling;

    int nblocks;
    if (rtc_memory_mode == isce3::core::MemoryModeBlocksY::SingleBlockY) {
        nblocks = 1;
        block_length_with_upsampling = imax;
        block_length = geogrid.length();
    } else {
        const int out_nbands = 1;
        getBlockProcessingParametersXY(
            imax, jmax, out_nbands, sizeof(T), &info,
                           &block_length_with_upsampling, &nblocks,
                           nullptr, nullptr, min_block_size, max_block_size,
                           geogrid_upsampling);
        block_length = block_length_with_upsampling / geogrid_upsampling;
    }

    info << "block length (with upsampling): " << block_length_with_upsampling
         << pyre::journal::endl;

    _Pragma("omp parallel for schedule(dynamic)")
        for (int block = 0; block < nblocks; ++block) {
            _RunBlock(jmax, block_length, block_length_with_upsampling, block,
                numdone, progress_block, geogrid_upsampling, interp_method,
                dem_raster, out_geo_rdr, out_geo_grid, start, pixazm, dr, r0,
                xbound, ybound, geogrid, radar_grid, input_dop, ellipsoid,
                orbit, threshold, num_iter, delta_range, out_gamma_array,
                out_beta_array, out_sigma_array, proj.get(), rtc_area_mode,
                rtc_area_beta_mode, input_terrain_radiometry,
                output_terrain_radiometry);
        }

    printf("\rRTC progress: 100%%\n");
    std::cout << std::endl;

    /*
    if (rtc_area_mode == rtcAreaMode::AREA_FACTOR && 
            rtc_area_beta_mode == rtcAreaBetaMode::PIXEL_AREA),
        the division by area beta (A_beta) is done using the "pixel area"
        computed by ground velocity and added to the variable `divisor`
        within _RunBlock()
    The lines below handle the pixel-wise division of A_beta using
    `out_beta_array`.
    */
    if (out_sigma != nullptr &&
            rtc_area_mode == rtcAreaMode::AREA_FACTOR &&
            rtc_area_beta_mode != rtcAreaBetaMode::PIXEL_AREA) {
        _normalizeRtcArea(out_sigma_array, out_beta_array, info); 
    }

    if (rtc_area_mode == rtcAreaMode::AREA_FACTOR &&
            rtc_area_beta_mode != rtcAreaBetaMode::PIXEL_AREA) {
        _normalizeRtcArea(out_gamma_array, out_beta_array, info); 
    }

    _applyRtcMinValueDb(out_gamma_array, rtc_min_value_db, rtc_area_mode, info);

    info << "saving RTC area normalization factor" << pyre::journal::endl;
    output_raster.setBlock(
            out_gamma_array.data(), 0, 0, radar_grid.width(), radar_grid.length());

    if (out_geo_rdr != nullptr) {
        double geotransform_edges[] = {
                geogrid.startX() - geogrid.spacingX() / 2.0,
                geogrid.spacingX() / geogrid_upsampling, 0,
                geogrid.startY() - geogrid.spacingY() / 2.0, 0,
                geogrid.spacingY() / geogrid_upsampling};
        out_geo_rdr->setGeoTransform(geotransform_edges);
        out_geo_rdr->setEPSG(geogrid.epsg());
    }

    if (out_geo_grid != nullptr) {
        double geotransform_grid[] = {geogrid.startX(),
                geogrid.spacingX() / geogrid_upsampling, 0, geogrid.startY(), 0,
                geogrid.spacingY() / geogrid_upsampling};
        out_geo_grid->setGeoTransform(geotransform_grid);
        out_geo_grid->setEPSG(geogrid.epsg());
    }

    if (out_sigma != nullptr) {
        info << "saving RTC area normalization factor to sigma0"
             << pyre::journal::endl;
        out_sigma->setBlock(out_sigma_array.data(), 0, 0, radar_grid.width(),
                radar_grid.length());
    }

    auto elapsed_time_milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time);
    float elapsed_time = ((float) elapsed_time_milliseconds.count()) / 1e3;
    info << "elapsed time (RTC-AP) [s]: " << elapsed_time
         << pyre::journal::endl;
}

/** Convert enum input terrain radiometry to string */
std::string get_input_terrain_radiometry_str(
        rtcInputTerrainRadiometry input_terrain_radiometry)
{
    std::string input_terrain_radiometry_str;
    switch (input_terrain_radiometry) {
    case rtcInputTerrainRadiometry::BETA_NAUGHT:
        input_terrain_radiometry_str = "beta-naught";
        break;
    case rtcInputTerrainRadiometry::SIGMA_NAUGHT_ELLIPSOID:
        input_terrain_radiometry_str = "sigma-naught";
        break;
    default:
        std::string error_message =
                "ERROR invalid input radiometric terrain radiometry";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_message);
    }
    return input_terrain_radiometry_str;
}

/** Convert enum output terrain radiometry to string */
std::string get_output_terrain_radiometry_str(
        rtcOutputTerrainRadiometry output_terrain_radiometry)
{
    std::string output_terrain_radiometry_str;
    switch (output_terrain_radiometry) {
    case rtcOutputTerrainRadiometry::SIGMA_NAUGHT:
        output_terrain_radiometry_str = "sigma-naught";
        break;
    case rtcOutputTerrainRadiometry::GAMMA_NAUGHT:
        output_terrain_radiometry_str = "gamma-naught";
        break;
    default:
        std::string error_message =
                "ERROR invalid output radiometric terrain radiometry";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_message);
    }
    return output_terrain_radiometry_str;
}

/** Convert enum rtc_area_mode to string */
std::string get_rtc_area_mode_str(rtcAreaMode rtc_area_mode)
{
    std::string rtc_area_mode_str;
    switch (rtc_area_mode) {
    case rtcAreaMode::AREA: rtc_area_mode_str = "area"; break;
    case rtcAreaMode::AREA_FACTOR:
        rtc_area_mode_str = "area normalization factor";
        break;
    default:
        std::string error_message = "ERROR invalid RTC area mode";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_message);
        break;
    }
    return rtc_area_mode_str;
}

/** Convert enum output_mode to string */
std::string get_rtc_area_beta_mode_str(rtcAreaBetaMode rtc_area_beta_mode)
{
    std::string rtc_area_beta_mode_str;
    switch (rtc_area_beta_mode) {
    case rtcAreaBetaMode::AUTO: rtc_area_beta_mode_str = "auto"; break;
    case rtcAreaBetaMode::PIXEL_AREA: rtc_area_beta_mode_str = "pixel area";
        break;
    case rtcAreaBetaMode::PROJECTION_ANGLE:
        rtc_area_beta_mode_str = "projection angle";
        break;
    default:
        std::string error_message = "ERROR invalid RTC area beta mode";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_message);
        break;
    }
    return rtc_area_beta_mode_str;
}


/** Convert enum output_mode to string */
std::string get_rtc_algorithm_str(rtcAlgorithm rtc_algorithm)
{
    std::string rtc_algorithm_str;
    switch (rtc_algorithm) {
    case rtcAlgorithm::RTC_BILINEAR_DISTRIBUTION:
        rtc_algorithm_str = "Bilinear distribution (D. Small))";
        break;
    case rtcAlgorithm::RTC_AREA_PROJECTION:
        rtc_algorithm_str = "Area projection";
        break;
    default:
        std::string error_message = "ERROR invalid RTC algorithm";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_message);
        break;
    }
    return rtc_algorithm_str;
}

void print_parameters(pyre::journal::info_t& channel,
        const isce3::product::RadarGridParameters& radar_grid,
        rtcInputTerrainRadiometry input_terrain_radiometry,
        rtcOutputTerrainRadiometry output_terrain_radiometry,
        rtcAreaMode rtc_area_mode, rtcAreaBetaMode rtc_area_beta_mode,
        double geogrid_upsampling,
        float rtc_min_value_db)
{
    std::string input_terrain_radiometry_str =
            get_input_terrain_radiometry_str(input_terrain_radiometry);

    std::string output_terrain_radiometry_str =
            get_output_terrain_radiometry_str(output_terrain_radiometry);

    std::string rtc_area_mode_str = get_rtc_area_mode_str(rtc_area_mode);

    std::string rtc_area_beta_mode_str = get_rtc_area_beta_mode_str(
        rtc_area_beta_mode);

    channel << "input radiometry: " << input_terrain_radiometry_str
            << pyre::journal::newline
            << "output radiometry: " << output_terrain_radiometry_str
            << pyre::journal::newline
            << "RTC area mode (area/area normalization factor): "
            << rtc_area_mode_str << pyre::journal::newline
            << "RTC area beta mode: "
            << rtc_area_beta_mode_str << pyre::journal::newline
            << "RTC geogrid upsampling: " << geogrid_upsampling
            << pyre::journal::newline << "look side: " << radar_grid.lookSide()
            << pyre::journal::newline
            << "radar-grid length: " << radar_grid.length()
            << ", width: " << radar_grid.width() << pyre::journal::newline
            << "RTC min value [dB]: " << rtc_min_value_db
            << pyre::journal::newline << pyre::journal::endl;
}
}} // namespace isce3::geometry
