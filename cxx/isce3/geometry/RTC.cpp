//-*- C++ -*-
//-*- coding: utf-8 -*-

#include "RTC.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <isce3/core/Constants.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Projections.h>
#include <isce3/error/ErrorCode.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/Geocode.h>
#include <isce3/geometry/boundingbox.h>
#include <isce3/geometry/geometry.h>
#include <isce3/product/RadarGridParameters.h>
#include <isce3/signal/Looks.h>
#include <string>

using isce::core::cartesian_t;
using isce::core::Mat3;
using isce::core::OrbitInterpBorderMode;
using isce::core::Vec3;

template<typename T1, typename T2>
auto operator*(const std::complex<T1>& lhs, const T2& rhs) {
    using U = typename std::common_type_t<T1, T2>;
    return std::complex<U>(lhs) * U(rhs);
}

template<typename T1, typename T2>
auto operator*(const T1& lhs, const std::complex<T2>& rhs) {
    using U = typename std::common_type_t<T1, T2>;
    return U(lhs) * std::complex<U>(rhs);
}

namespace isce {
namespace geometry {

int _omp_thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

template <typename T> struct is_complex_t : std::false_type {};
template <typename T> struct is_complex_t<std::complex<T>> : std::true_type {};
template <typename T>
constexpr bool is_complex() { return is_complex_t<T>::value; }

template<typename T>
void _applyRTC(isce::io::Raster& input_raster, isce::io::Raster& input_rtc,
               isce::io::Raster& output_raster, float rtc_min_value,
               double abs_cal_factor, pyre::journal::info_t& info,
               bool flag_complex_to_real_squared) {

    int nbands = input_raster.numBands();
    int width = input_raster.width();
    int length = input_raster.length();

    int min_block_length = 32;
    int block_size;
    int nblocks = areaProjGetNBlocks(length, &info, 0,
                                     &block_size, nullptr, min_block_length);

    if (std::isnan(rtc_min_value))
        rtc_min_value = 0;

    // for each band in the input:
    for (size_t band = 0; band < nbands; ++band) {
        info << "applying RTC to band: " << band + 1 << "/" << nbands
             << pyre::journal::endl;

        // get a block of data
        #pragma omp parallel for schedule(dynamic)
        for (int block = 0; block < nblocks; ++block) {

            int effective_block_size = block_size;
            if (block * block_size + effective_block_size > length - 1) {
                effective_block_size = length - block * block_size;
            }

            isce::core::Matrix<float> rtc_ratio(effective_block_size, width);
            #pragma omp critical
            {
                input_rtc.getBlock(rtc_ratio.data(), 0, block * block_size,
                                   width, effective_block_size, 1);
            }

            isce::core::Matrix<T> radar_data_block(block_size, width);
            if (!flag_complex_to_real_squared) {
                #pragma omp critical
                {
                    input_raster.getBlock(radar_data_block.data(), 0,
                                          block * block_size, width,
                                          effective_block_size, band + 1);
                }
                for (int i = 0; i < effective_block_size; ++i)
                    for (int jj = 0; jj < width; ++jj) {
                        float rtc_ratio_value = rtc_ratio(i, jj);
                        if (!std::isnan(rtc_ratio_value) &&
                                rtc_ratio_value > rtc_min_value) {
                            float factor = abs_cal_factor / rtc_ratio_value;

                            if (is_complex_t<T>())

                                factor = std::sqrt(factor);
                            radar_data_block(i, jj) *= factor;
                        } else {
                            radar_data_block(i, jj) =
                                    std::numeric_limits<float>::quiet_NaN();
                        }
                    }
            } else {
                isce::core::Matrix<std::complex<T>> radar_data_block_complex(
                        block_size, width);
                #pragma omp critical
                {
                    input_raster.getBlock(radar_data_block_complex.data(), 0,
                                          block * block_size, width,
                                          effective_block_size, band + 1);
                }
                for (int i = 0; i < effective_block_size; ++i)
                    for (int jj = 0; jj < width; ++jj) {
                        float rtc_ratio_value = rtc_ratio(i, jj);
                        if (!std::isnan(rtc_ratio_value) &&
                            rtc_ratio_value > rtc_min_value) {
                            float factor = abs_cal_factor / rtc_ratio_value;
                            std::complex<T> radar_data_complex =
                                    radar_data_block_complex(i, jj);
                            radar_data_block(i, jj) =
                                    (radar_data_complex.real() *
                                             radar_data_complex.real() +
                                     radar_data_complex.imag() *
                                             radar_data_complex.imag());
                            radar_data_block(i, jj) *= factor;
                        } else
                            radar_data_block(i, jj) =
                                    std::numeric_limits<float>::quiet_NaN();
                    }
            }

// set output
#pragma omp critical
            {
                output_raster.setBlock(radar_data_block.data(), 0,
                                       block * block_size, width,
                                       effective_block_size, band + 1);
            }
        }
    }
}

void applyRTC(const isce::product::RadarGridParameters& radar_grid,
              const isce::core::Orbit& orbit,
              const isce::core::LUT2d<double>& input_dop,
              isce::io::Raster& input_raster, isce::io::Raster& dem_raster,
              isce::io::Raster& output_raster,
              rtcInputRadiometry input_radiometry, int exponent,
              rtcAreaMode rtc_area_mode, rtcAlgorithm rtc_algorithm,
              double geogrid_upsampling, float rtc_min_value_db,
              double abs_cal_factor, float radar_grid_nlooks,
              isce::io::Raster* out_nlooks, isce::io::Raster* input_rtc,
              isce::io::Raster* output_rtc, rtcMemoryMode rtc_memory_mode) {

    if (exponent < 0 || exponent > 2) {
        std::string error_message =
                "ERROR invalid exponent for RTC pre-process. Valid options:";
        error_message += " 0 (auto selection), 1 (linear), or 2 (square).";
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), error_message);
    }

    bool flag_complex_to_real = isce::signal::verifyComplexToRealCasting(
            input_raster, output_raster, exponent);

    pyre::journal::info_t info("isce.geometry.applyRTC");

    // declare pointer to the raster containing the RTC area factor
    isce::io::Raster* rtc_raster;
    std::unique_ptr<isce::io::Raster> rtc_raster_unique_ptr;

    if (input_rtc == nullptr) {

        // if RTC (area factor) raster does not needed to be saved,
        // initialize it as a GDAL memory virtual file
        if (output_rtc == nullptr) {
            rtc_raster_unique_ptr = std::make_unique<isce::io::Raster>(
                    "/vsimem/dummy", radar_grid.width(), radar_grid.length(), 1,
                    GDT_Float32, "ENVI");
            rtc_raster = rtc_raster_unique_ptr.get();
        }

        // Otherwise, copies the pointer to the output RTC file
        else
            rtc_raster = output_rtc;

        info << "calculating RTC..." << pyre::journal::endl;
        facetRTC(radar_grid, orbit, input_dop, dem_raster, *rtc_raster,
                 input_radiometry, rtc_area_mode, rtc_algorithm,
                 geogrid_upsampling, rtc_min_value_db, radar_grid_nlooks,
                 out_nlooks, rtc_memory_mode);
    } else {
        info << "reading pre-computed RTC..." << pyre::journal::endl;
        rtc_raster = input_rtc;
    }

    if (abs_cal_factor != 1)
        info << "absolute calibration factor: " << abs_cal_factor
            << pyre::journal::endl;

    float rtc_min_value = 0;
    if (!std::isnan(rtc_min_value_db)) {
        rtc_min_value = std::pow(10, (rtc_min_value_db / 10));
        info << "applying min. RTC value: " << rtc_min_value_db
             << " [dB] = " << rtc_min_value << pyre::journal::endl;
    }

    if (input_raster.dtype() == GDT_Float32 ||
        (input_raster.dtype() == GDT_CFloat32 && flag_complex_to_real))
        _applyRTC<float>(input_raster, *rtc_raster, output_raster,
                         rtc_min_value_db, abs_cal_factor,
                         info, flag_complex_to_real);
    else if (input_raster.dtype() == GDT_Float64 ||
             (input_raster.dtype() == GDT_CFloat64 && flag_complex_to_real))
        _applyRTC<double>(input_raster, *rtc_raster, output_raster,
                          rtc_min_value_db, abs_cal_factor,
                          info, flag_complex_to_real);
    else if (input_raster.dtype() == GDT_CFloat32)
        _applyRTC<std::complex<float>>(
                input_raster, *rtc_raster, output_raster, rtc_min_value_db,
                abs_cal_factor, info, flag_complex_to_real);
    else if (input_raster.dtype() == GDT_CFloat64)
        _applyRTC<std::complex<double>>(
                input_raster, *rtc_raster, output_raster, rtc_min_value_db,
                abs_cal_factor, info, flag_complex_to_real);
    else {
        std::string error_message =
                "ERROR not implemented for input raster datatype";
        throw isce::except::RuntimeError(ISCE_SRCINFO(), error_message);
    }
}

double
computeUpsamplingFactor(const DEMInterpolator& dem_interp,
                        const isce::product::RadarGridParameters& radar_grid,
                        const isce::core::Ellipsoid& ellps) {

    // Create a projection object from the DEM interpolator
    std::unique_ptr<isce::core::ProjectionBase> proj(
            isce::core::createProj(dem_interp.epsgCode()));

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

int areaProjGetNBlocks(int array_length, pyre::journal::info_t* channel,
                       int upsampling, 
                       int* block_length_with_upsampling, 
                       int* block_length,
                       int min_block_length, int max_block_length)
{

    auto n_threads = _omp_thread_count();
    auto nblocks = 4 * n_threads;
    auto min_block_length_eff = std::min(array_length, min_block_length);

    // limiting block size
    int _block_length_with_upsampling = std::min(
            max_block_length,
            std::max(min_block_length_eff,
                     (int) std::ceil(((float) array_length) / nblocks)));

    int _block_length = _block_length_with_upsampling;

    // snap (_block_length_with_upsampling multiple of upsampling)
    if (upsampling > 0) {
        _block_length = _block_length_with_upsampling / upsampling;
        _block_length_with_upsampling = _block_length * upsampling;
    }

    nblocks = std::ceil(((float) array_length) / _block_length_with_upsampling);

    if (channel != nullptr) {
        *channel << "array length: " << array_length << pyre::journal::endl;
        *channel << "number of available thread(s): " << n_threads << pyre::journal::endl;
        *channel << "number of block(s): " << nblocks << pyre::journal::endl;
    }
    
    
    if (block_length != nullptr) {
        *block_length = _block_length;
        if (channel != nullptr) {
            *channel << "block length (without upsampling): " << *block_length
                     << pyre::journal::endl;
        }
    }
    if (block_length_with_upsampling != nullptr) {
        *block_length_with_upsampling = _block_length_with_upsampling;
        *channel << "block length: "
                 << *block_length_with_upsampling << pyre::journal::endl;
    }

    return nblocks;
}

void facetRTC(isce::product::Product& product, isce::io::Raster& dem_raster,
              isce::io::Raster& output_raster, char frequency,
              bool native_doppler, rtcInputRadiometry input_radiometry,
              rtcAreaMode rtc_area_mode, rtcAlgorithm rtc_algorithm,
              double geogrid_upsampling, float rtc_min_value_db,
              size_t nlooks_az, size_t nlooks_rg, isce::io::Raster* out_nlooks,
              rtcMemoryMode rtc_memory_mode) {

    isce::core::Orbit orbit = product.metadata().orbit();
    isce::product::RadarGridParameters radar_grid(product, frequency);
    isce::product::RadarGridParameters radar_grid_ml =
            radar_grid.multilook(nlooks_az, nlooks_rg);

    // Get a copy of the Doppler LUT; allow for out-of-bounds extrapolation
    isce::core::LUT2d<double> dop;
    if (native_doppler)
        dop = product.metadata().procInfo().dopplerCentroid(frequency);

    int radar_grid_nlooks = nlooks_az * nlooks_rg;

    facetRTC(radar_grid_ml, orbit, dop, dem_raster, output_raster,
             input_radiometry, rtc_area_mode, rtc_algorithm, geogrid_upsampling,
             rtc_min_value_db, radar_grid_nlooks, out_nlooks, rtc_memory_mode);
}

void facetRTC(const isce::product::RadarGridParameters& radar_grid,
              const isce::core::Orbit& orbit,
              const isce::core::LUT2d<double>& input_dop,
              isce::io::Raster& dem_raster, isce::io::Raster& output_raster,
              rtcInputRadiometry input_radiometry, rtcAreaMode rtc_area_mode,
              rtcAlgorithm rtc_algorithm, double geogrid_upsampling,
              float rtc_min_value_db, float radar_grid_nlooks,
              isce::io::Raster* out_nlooks, rtcMemoryMode rtc_memory_mode,
              isce::core::dataInterpMethod interp_method, double threshold,
              int num_iter, double delta_range) {

    double geotransform[6];
    dem_raster.getGeoTransform(geotransform);
    const double dy = geotransform[5];
    const double dx = geotransform[1];

    int epsg = dem_raster.getEPSG();
    std::unique_ptr<isce::core::ProjectionBase> proj(
            isce::core::createProj(epsg));

    BoundingBox bbox = getGeoBoundingBoxHeightSearch(radar_grid, orbit,
                                                     proj.get(), input_dop);

    const int MARGIN_PIXELS = 20;
    double y0 = bbox.MinY - MARGIN_PIXELS * std::abs(dy);
    double yf = bbox.MaxY + MARGIN_PIXELS * std::abs(dy);
    const double x0 = bbox.MinX - MARGIN_PIXELS * dx;
    const double xf = bbox.MaxX + MARGIN_PIXELS * dx;
    const int geogrid_length = std::abs(std::ceil((yf - y0) / dy));
    const int geogrid_width = std::ceil((xf - x0) / dx);

    if (dy < 0)
        std::swap(y0, yf);

    facetRTC(dem_raster, output_raster, radar_grid, orbit, input_dop, y0, dy,
             x0, dx, geogrid_length, geogrid_width, epsg, input_radiometry,
             rtc_area_mode, rtc_algorithm, geogrid_upsampling, rtc_min_value_db,
             radar_grid_nlooks, nullptr, nullptr, out_nlooks, rtc_memory_mode,
             interp_method, threshold, num_iter, delta_range);
}

void facetRTC(isce::io::Raster& dem_raster, isce::io::Raster& output_raster,
              const isce::product::RadarGridParameters& radar_grid,
              const isce::core::Orbit& orbit,
              const isce::core::LUT2d<double>& input_dop, const double y0,
              const double dy, const double x0, const double dx,
              const int geogrid_length, const int geogrid_width, const int epsg,
              rtcInputRadiometry input_radiometry, rtcAreaMode rtc_area_mode,
              rtcAlgorithm rtc_algorithm, double geogrid_upsampling,
              float rtc_min_value_db, float radar_grid_nlooks,
              isce::io::Raster* out_geo_vertices,
              isce::io::Raster* out_geo_grid, isce::io::Raster* out_nlooks,
              rtcMemoryMode rtc_memory_mode,
              isce::core::dataInterpMethod interp_method, double threshold,
              int num_iter, double delta_range)
{

    if (rtc_algorithm == rtcAlgorithm::RTC_AREA_PROJECTION) {
        facetRTCAreaProj(
                dem_raster, output_raster, radar_grid, orbit, input_dop, y0, dy,
                x0, dx, geogrid_length, geogrid_width, epsg, input_radiometry,
                rtc_area_mode, geogrid_upsampling, rtc_min_value_db,
                radar_grid_nlooks, out_geo_vertices, out_geo_grid, out_nlooks,
                rtc_memory_mode, interp_method, threshold, num_iter, 
                delta_range);
    } else {
        facetRTCDavidSmall(dem_raster, output_raster, radar_grid, orbit,
                           input_dop, y0, dy, x0, dx, geogrid_length,
                           geogrid_width, epsg, input_radiometry, rtc_area_mode,
                           geogrid_upsampling);
    }
}

void areaProjIntegrateSegment(double y1, double y2, double x1, double x2,
                              int length, int width,
                              isce::core::Matrix<double>& w_arr, double& nlooks,
                              int plane_orientation)
{

    // if line is vertical or out of boundaries, return
    if (x2 == x1 || (x1 < 0 && x2 < 0) ||
        (x1 >= width - 1 && x2 >= width - 1) || (y1 < 0 && y2 < 0) ||
        (y1 >= length - 1 && y2 >= length - 1))
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
        const double y_start_next = slope * (x_start + x_increment_margin) + offset;
        const int x_index = std::floor(x_start);
        int y_index = std::floor(y_start_next);

        if (y_index < 0 ) {
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
            x_next = std::min((y_index + 1 - offset) / slope,
                              (double) x_index + 1);
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

void _addArea(double area, isce::core::Matrix<float>& out_array,
              float radar_grid_nlooks,
              isce::core::Matrix<float>& out_nlooks_array, int length,
              int width, int x_min, int y_min, int size_x, int size_y,
              isce::core::Matrix<double>& w_arr, double nlooks,
              isce::core::Matrix<double>& w_arr_out, double& nlooks_out,
              double x_center, double x_left, double x_right, double y_center,
              double y_left, double y_right, int plane_orientation) {
    areaProjIntegrateSegment(y_left, y_right, x_left, x_right, size_y, size_x,
                             w_arr, nlooks, plane_orientation);

    nlooks_out = 0;
    areaProjIntegrateSegment(y_center, y_right, x_center, x_right, size_y,
                             size_x, w_arr_out, nlooks_out, plane_orientation);

    for (int ii = 0; ii < size_y; ++ii)
        for (int jj = 0; jj < size_x; ++jj) {
            double w = w_arr(ii, jj) - w_arr_out(ii, jj);
            w_arr(ii, jj) = 0;
            if (w == 0 || w * area < 0)
                continue;
            int y = ii + y_min;
            int x = jj + x_min;
            if (x < 0 || y < 0 || y >= length || x >= width)
                continue;
            if (out_nlooks_array.data() != nullptr) {
                const auto out_nlooks = radar_grid_nlooks * std::abs(w * (nlooks - nlooks_out));
                _Pragma("omp atomic")
                out_nlooks_array(y, x) += out_nlooks;
            }
            w /= nlooks - nlooks_out;
            _Pragma("omp atomic")
            out_array(y, x) += w * area;
        }
}

double computeFacet(Vec3 xyz_center, Vec3 xyz_left, Vec3 xyz_right,
                    Vec3 lookXYZ, double p1, double& p3, double divisor,
                    bool clockwise_direction) {
    const Vec3 normal_facet = normalPlane(xyz_center, xyz_left, xyz_right);
    double cos_inc_facet = normal_facet.dot(lookXYZ);

    p3 = (xyz_center - xyz_right).norm();

    if (clockwise_direction)
        cos_inc_facet *= -1;

    // If facet is not illuminated by radar, skip
    if (cos_inc_facet <= 0)
        return 0;

    // Side lengths (keep p3 solution for next iteration)
    const double p2 = (xyz_right - xyz_left).norm();

    // Heron's formula to get area of facets in XYZ coordinates
    const float h = 0.5 * (p1 + p2 + p3);
    float area = cos_inc_facet * std::sqrt(h * (h - p1) * (h - p2) * (h - p3)) /
                 divisor;

    return area;
}

void facetRTCDavidSmall(isce::io::Raster& dem_raster,
                        isce::io::Raster& output_raster,
                        const isce::product::RadarGridParameters& radar_grid,
                        const isce::core::Orbit& orbit,
                        const isce::core::LUT2d<double>& input_dop,
                        const double y0, const double dy, const double x0,
                        const double dx, const int geogrid_length,
                        const int geogrid_width, const int epsg,
                        rtcInputRadiometry input_radiometry,
                        rtcAreaMode rtc_area_mode, double upsample_factor) {

    pyre::journal::info_t info("isce.geometry.facetRTCDavidSmall");

    std::unique_ptr<isce::core::ProjectionBase> proj(
            isce::core::createProj(epsg));
    const isce::core::Ellipsoid& ellps = proj->ellipsoid();

    print_parameters(info, radar_grid, y0, dy, x0, dx, geogrid_length,
                     geogrid_width, input_radiometry, rtc_area_mode,
                     upsample_factor);

    const double yf = y0 + geogrid_length * dy;
    const double margin_x = std::abs(dx) * 20;
    const double margin_y = std::abs(dy) * 20;

    DEMInterpolator dem_interp(0,
                               isce::core::dataInterpMethod::BIQUINTIC_METHOD);
    dem_interp.loadDEM(
            dem_raster, x0 - margin_x, x0 + geogrid_width * dx + margin_x,
            std::min(y0, yf) - margin_y, std::max(y0, yf) + margin_y);

    const double start = radar_grid.sensingStart();
    const double pixazm =
            radar_grid.azimuthTimeInterval(); // azimuth difference per pixel

    const double r0 = radar_grid.startingRange();
    const double dr = radar_grid.rangePixelSpacing();

    // Bounds for valid RDC coordinates
    double xbound = radar_grid.width() - 1.0;
    double ybound = radar_grid.length() - 1.0;

    // Output raster
    isce::core::Matrix<float> out(radar_grid.length(), radar_grid.width());
    out.fill(0);

    // ------------------------------------------------------------------------
    // Main code: decompose DEM into facets, compute RDC coordinates
    // ------------------------------------------------------------------------

    // Enter loop to read in SLC range/azimuth coordinates and compute area
    std::cout << std::endl;

    if (std::isnan(upsample_factor))
        upsample_factor =
                computeUpsamplingFactor(dem_interp, radar_grid, ellps);

    const size_t imax = geogrid_length * upsample_factor;
    const size_t jmax = geogrid_width * upsample_factor;

    const size_t progress_block = imax * jmax / 100;
    size_t numdone = 0;
    auto side = radar_grid.lookSide();

// Loop over DEM facets
#pragma omp parallel for schedule(dynamic)
    for (size_t ii = 0; ii < imax; ++ii) {
        double a = radar_grid.sensingMid();
        double r = radar_grid.midRange();

        // The inner loop is not parallelized in order to keep the previous
        // solution from geo2rdr within the same thread. This solution is used
        // as the initial guess for the next call to geo2rdr.
        for (size_t jj = 0; jj < jmax; ++jj) {
#pragma omp atomic
            numdone++;

            if (numdone % progress_block == 0)
#pragma omp critical
                printf("\rRTC progress: %d%%",
                       (int) (numdone * 1e2 / (imax * jmax))),
                        fflush(stdout);
            // Central DEM coordinates of facets
            const double dem_ymid = y0 + dy * (0.5 + ii) / upsample_factor;
            const double dem_xmid = x0 + dx * (0.5 + jj) / upsample_factor;

            const Vec3 inputDEM {dem_xmid, dem_ymid,
                                 dem_interp.interpolateXY(dem_xmid, dem_ymid)};
            // Compute facet-central LLH vector
            const Vec3 inputLLH = proj->inverse(inputDEM);
            // Should incorporate check on return status here
            int converged =
                    geo2rdr(inputLLH, ellps, orbit, input_dop, a, r,
                            radar_grid.wavelength(), side, 1e-4, 100, 1e-4);
            if (!converged)
                continue;

            const float azpix = (a - start) / pixazm;
            const float ranpix = (r - r0) / dr;

            // Establish bounds for bilinear weighting model
            const int x1 = (int) std::floor(ranpix);
            const int x2 = x1 + 1;
            const int y1 = (int) std::floor(azpix);
            const int y2 = y1 + 1;

            // Check to see if pixel lies in valid RDC range
            if (ranpix < -1 or x2 > xbound + 1 or azpix < -1 or y2 > ybound + 1)
                continue;

            // Current x/y-coords in DEM
            const double dem_y0 = y0 + dy * ii / upsample_factor;
            const double dem_y1 = dem_y0 + dy / upsample_factor;
            const double dem_x0 = x0 + dx * jj / upsample_factor;
            const double dem_x1 = dem_x0 + dx / upsample_factor;

            // Set DEM-coordinate corner vectors
            const Vec3 dem00 = {dem_x0, dem_y0,
                                dem_interp.interpolateXY(dem_x0, dem_y0)};
            const Vec3 dem01 = {dem_x0, dem_y1,
                                dem_interp.interpolateXY(dem_x0, dem_y1)};
            const Vec3 dem10 = {dem_x1, dem_y0,
                                dem_interp.interpolateXY(dem_x1, dem_y0)};
            const Vec3 dem11 = {dem_x1, dem_y1,
                                dem_interp.interpolateXY(dem_x1, dem_y1)};

            // Convert to XYZ
            const Vec3 xyz00 = ellps.lonLatToXyz(proj->inverse(dem00));
            const Vec3 xyz01 = ellps.lonLatToXyz(proj->inverse(dem01));
            const Vec3 xyz10 = ellps.lonLatToXyz(proj->inverse(dem10));
            const Vec3 xyz11 = ellps.lonLatToXyz(proj->inverse(dem11));

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
            const float AP1 = std::sqrt(h1 * (h1 - p00_01) * (h1 - p00_10) *
                                        (h1 - p10_01));
            const float AP2 = std::sqrt(h2 * (h2 - p11_01) * (h2 - p11_10) *
                                        (h2 - p10_01));

            // Compute look angle from sensor to ground
            const Vec3 xyz_mid = ellps.lonLatToXyz(inputLLH);
            isce::core::cartesian_t xyz_plat, vel;
            isce::error::ErrorCode status = orbit.interpolate(
                    &xyz_plat, &vel, a, OrbitInterpBorderMode::FillNaN);
            if (status != isce::error::ErrorCode::Success)
                continue;

            const Vec3 lookXYZ = (xyz_plat - xyz_mid).normalized();

            // Compute dot product between each facet and look vector
            double cos_inc_facet_1 = lookXYZ.dot(normal_facet_1);
            double cos_inc_facet_2 = lookXYZ.dot(normal_facet_2);
            if (dy < 0) {
                cos_inc_facet_1 *= -1;
                cos_inc_facet_2 *= -1;
            }

            // If facets are not illuminated by radar, skip
            if (cos_inc_facet_1 <= 0. and cos_inc_facet_2 <= 0.)
                continue;

            // Compute projected area
            float area = 0;
            if (cos_inc_facet_1 > 0)
                area += AP1 * cos_inc_facet_1;
            if (cos_inc_facet_2 > 0)
                area += AP2 * cos_inc_facet_2;
            if (area == 0)
                continue;

            // Compute fractional weights from indices
            const float Wr = ranpix - x1;
            const float Wa = azpix - y1;
            const float Wrc = 1. - Wr;
            const float Wac = 1. - Wa;

            if (rtc_area_mode == rtcAreaMode::AREA_FACTOR) {
                const double area_beta = radar_grid.rangePixelSpacing() *
                                         vel.norm() / radar_grid.prf();
                area /= area_beta;
            }

            // if if (ranpix < -1 or x2 > xbound+1 or azpix < -1 or y2 >
            // ybound+1)
            if (y1 >= 0 && x1 >= 0) {
#pragma omp atomic
                out(y1, x1) += area * Wrc * Wac;
            }
            if (y1 >= 0 && x2 <= xbound) {
#pragma omp atomic
                out(y1, x2) += area * Wr * Wac;
            }
            if (y2 <= ybound && x1 >= 0) {
#pragma omp atomic
                out(y2, x1) += area * Wrc * Wa;
            }
            if (y2 <= ybound && x2 <= xbound) {
#pragma omp atomic
                out(y2, x2) += area * Wr * Wa;
            }
        }
    }

    printf("\rRTC progress: 100%%");
    std::cout << std::endl;

    float max_hgt, avg_hgt;

    dem_interp.computeHeightStats(max_hgt, avg_hgt, info);
    DEMInterpolator flat_interp(avg_hgt);

    if (input_radiometry == rtcInputRadiometry::SIGMA_NAUGHT_ELLIPSOID) {
        // Compute the flat earth incidence angle correction
        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (size_t i = 0; i < radar_grid.length(); ++i) {
            for (size_t j = 0; j < radar_grid.width(); ++j) {

                isce::core::cartesian_t xyz_plat, vel;
                double a = start + i * pixazm;
                isce::error::ErrorCode status = orbit.interpolate(
                        &xyz_plat, &vel, a, OrbitInterpBorderMode::FillNaN);
                if (status != isce::error::ErrorCode::Success)
                    continue;

                // Slant range for current pixel
                const double slt_range = r0 + j * dr;

                // Get LLH and XYZ coordinates for this azimuth/range
                isce::core::cartesian_t targetLLH, targetXYZ;
                targetLLH[2] = avg_hgt; // initialize first guess
                rdr2geo(a, slt_range, 0, orbit, ellps, flat_interp, targetLLH,
                        radar_grid.wavelength(), side, 1e-4, 20, 20);

                // Computation of ENU coordinates around ground target
                ellps.lonLatToXyz(targetLLH, targetXYZ);
                const Vec3 satToGround = targetXYZ - xyz_plat;
                const Mat3 xyz2enu = Mat3::xyzToEnu(targetLLH[1], targetLLH[0]);
                const Vec3 enu = xyz2enu.dot(satToGround);

                // Compute incidence angle components
                const double costheta = std::abs(enu[2]) / enu.norm();
                const double sintheta = std::sqrt(1. - costheta * costheta);

                out(i, j) *= sintheta;
            }
        }
    }

    output_raster.setBlock(out.data(), 0, 0, radar_grid.width(),
                           radar_grid.length());
}

void _RunBlock(const int jmax, int block_size, int block_size_with_upsampling,
               int block, int& numdone, int progress_block,
               double geogrid_upsampling,
               isce::core::dataInterpMethod interp_method,
               isce::io::Raster& dem_raster, isce::io::Raster* out_geo_vertices,
               isce::io::Raster* out_geo_grid, const double start,
               const double pixazm, const double dr, double r0, int xbound,
               int ybound, const double y0, const double dy, const double x0,
               const double dx, const int geogrid_length,
               const int geogrid_width,
               const isce::product::RadarGridParameters& radar_grid,
               const isce::core::LUT2d<double>& dop,
               const isce::core::Ellipsoid& ellipsoid,
               const isce::core::Orbit& orbit, double threshold, int num_iter,
               double delta_range, isce::core::Matrix<float>& out_array,
               isce::core::Matrix<float>& out_nlooks_array,
               isce::core::ProjectionBase* proj, rtcAreaMode rtc_area_mode,
               rtcInputRadiometry input_radiometry, float radar_grid_nlooks) {

    auto side = radar_grid.lookSide();

    int this_block_size = block_size;
    if ((block + 1) * block_size > geogrid_length)
        this_block_size = geogrid_length % block_size;

    const int this_block_size_with_upsampling =
            this_block_size * geogrid_upsampling;
    int ii_0 = block * block_size_with_upsampling;

    DEMInterpolator dem_interp_block(0, interp_method);

    isce::core::Matrix<float> out_geo_vertices_a;
    isce::core::Matrix<float> out_geo_vertices_r;
    if (out_geo_vertices != nullptr) {
        out_geo_vertices_a.resize(this_block_size_with_upsampling + 1,
                                  jmax + 1);
        out_geo_vertices_r.resize(this_block_size_with_upsampling + 1,
                                  jmax + 1);
        out_geo_vertices_a.fill(std::numeric_limits<float>::quiet_NaN());
        out_geo_vertices_r.fill(std::numeric_limits<float>::quiet_NaN());
    }

    isce::core::Matrix<float> out_geo_grid_a;
    isce::core::Matrix<float> out_geo_grid_r;

    if (out_geo_grid != nullptr) {
        out_geo_grid_r.resize(this_block_size_with_upsampling, jmax);
        out_geo_grid_a.resize(this_block_size_with_upsampling, jmax);
        out_geo_grid_r.fill(std::numeric_limits<float>::quiet_NaN());
        out_geo_grid_a.fill(std::numeric_limits<float>::quiet_NaN());
    }

    // Convert margin to meters it not LonLat
    const double minX = x0;
    const double maxX = x0 + dx * geogrid_width;
    double minY = y0 + (dy * ii_0) / geogrid_upsampling;
    double maxY = y0 + (dy * (ii_0 + this_block_size_with_upsampling)) /
                               geogrid_upsampling;

    const double margin_x = std::abs(dx) * 20;
    const double margin_y = std::abs(dy) * 20;

#pragma omp critical
    {
        dem_interp_block.loadDEM(dem_raster, minX - margin_x, maxX + margin_x,
                                 std::min(minY, maxY) - margin_y,
                                 std::max(minY, maxY) + margin_y);
    }

    double a11 = radar_grid.sensingMid(), r11 = radar_grid.midRange();
    Vec3 dem11;

    std::vector<double> a_last(jmax + 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_last(jmax + 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_last(jmax + 1,
                               {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()});

    double dem_y1 = y0 + (dy * ii_0) / geogrid_upsampling;

    for (int jj = 0; jj <= jmax; ++jj) {
        const double dem_x1 = x0 + (dx * jj) / geogrid_upsampling;
        dem11 = {dem_x1, dem_y1,
                 dem_interp_block.interpolateXY(dem_x1, dem_y1)};
        int converged = geo2rdr(proj->inverse(dem11), ellipsoid, orbit, dop,
                                a11, r11, radar_grid.wavelength(), side,
                                threshold, num_iter, delta_range);
        if (!converged) {
            a11 = radar_grid.sensingMid();
            r11 = radar_grid.midRange();
            continue;
        }

        a_last[jj] = a11;
        r_last[jj] = r11;
        dem_last[jj] = dem11;
    }

    a11 = radar_grid.sensingMid();
    r11 = radar_grid.midRange();

    for (int i = 0; i < this_block_size_with_upsampling; ++i) {

        int ii = block * block_size_with_upsampling + i;

        if (!std::isnan(a_last[0])) {
            a11 = a_last[0];
            r11 = r_last[0];
        } else if (!std::isnan(a_last[1])) {
            a11 = a_last[1];
            r11 = r_last[1];
        }

        const double dem_x1_0 = x0;
        const double dem_y1 = y0 + dy * (1.0 + ii) / geogrid_upsampling;
        dem11 = {dem_x1_0, dem_y1,
                 dem_interp_block.interpolateXY(dem_x1_0, dem_y1)};
        int converged = geo2rdr(proj->inverse(dem11), ellipsoid, orbit, dop,
                                a11, r11, radar_grid.wavelength(), side,
                                threshold, num_iter, delta_range);
        if (!converged) {
            a11 = std::numeric_limits<double>::quiet_NaN();
            r11 = std::numeric_limits<double>::quiet_NaN();
        }

        for (int jj = 0; jj < (int) jmax; ++jj) {

#pragma omp atomic
            numdone++;
            if (numdone % progress_block == 0)
#pragma omp critical
                printf("\rRTC progress: %d%%", (int) numdone / progress_block),
                        fflush(stdout);

            // bottom left (copy from previous bottom right)
            const double a10 = a11;
            const double r10 = r11;
            const Vec3 dem10 = dem11;

            // top left
            const double a00 = a_last[jj];
            const double r00 = r_last[jj];
            const Vec3 dem00 = dem_last[jj];

            // top right
            const double a01 = a_last[jj + 1];
            const double r01 = r_last[jj + 1];
            const Vec3 dem01 = dem_last[jj + 1];

            // update "last" arrays (from lower left vertex)
            if (!std::isnan(a10)) {
                a_last[jj] = a10;
                r_last[jj] = r10;
                dem_last[jj] = dem10;
            }

            // calculate new bottom right
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

            const double dem_x1 = x0 + dx * (1.0 + jj) / geogrid_upsampling;
            dem11 = {dem_x1, dem_y1,
                     dem_interp_block.interpolateXY(dem_x1, dem_y1)};
            int converged = geo2rdr(proj->inverse(dem11), ellipsoid, orbit, dop,
                                    a11, r11, radar_grid.wavelength(), side,
                                    threshold, num_iter, delta_range);
            if (!converged) {
                a11 = std::numeric_limits<double>::quiet_NaN();
                r11 = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            // if last column also update top-right "last" arrays (from lower
            //   right vertex)
            if (jj == jmax - 1) {
                a_last[jj+1] = a11;
                r_last[jj+1] = r11;
                dem_last[jj+1] = dem11;
            }

            // define slant-range window
            int y_min = std::floor((std::min(std::min(a00, a01),
                                             std::min(a10, a11)) -
                                    start) /
                                   pixazm) -
                        1;
            if (y_min < -isce::core::AREA_PROJECTION_RADAR_GRID_MARGIN || 
                    y_min > ybound + 1)
                continue;
            int x_min = std::floor((std::min(std::min(r00, r01),
                                             std::min(r10, r11)) -
                                    r0) /
                                   dr) -
                        1;
            if (x_min < -isce::core::AREA_PROJECTION_RADAR_GRID_MARGIN ||
                    x_min > xbound + 1)
                continue;
            int y_max = std::ceil((std::max(std::max(a00, a01),
                                            std::max(a10, a11)) -
                                   start) /
                                  pixazm) +
                        1;
            if (y_max > ybound + 1 + 
                isce::core::AREA_PROJECTION_RADAR_GRID_MARGIN || 
                    y_max < -1 || y_max < y_min)
                continue;
            int x_max = std::ceil((std::max(std::max(r00, r01),
                                            std::max(r10, r11)) -
                                   r0) /
                                  dr) +
                        1;
            if (x_max > xbound + 1 +
                isce::core::AREA_PROJECTION_RADAR_GRID_MARGIN || 
                    x_max < -1 || x_max < x_min)
                continue;

            if (out_geo_vertices != nullptr)
            {
                if (i == 0) {
                    out_geo_vertices_a(i, jj + 1) = (a01 - start) / pixazm;
                    out_geo_vertices_r(i, jj + 1) = (a01 - r0) / dr;
                }
                if (i == 0 && jj == 0) {
                    out_geo_vertices_a(i, jj) = (a00 - start) / pixazm;
                    out_geo_vertices_r(i, jj) = (r00 - r0) / dr;
                }
                if (jj == 0) {
                    out_geo_vertices_a((i + 1), jj) = (a10 - start) / pixazm;
                    out_geo_vertices_r((i + 1), jj) = (r10 - r0) / dr;
                }
                out_geo_vertices_a((i + 1), jj + 1) = (a11 - start) / pixazm;
                out_geo_vertices_r((i + 1), jj + 1) = (r11 - r0) / dr;
            }

            if (std::isnan(a10) || std::isnan(a11) || std::isnan(a01) ||
                std::isnan(a00))
                continue;

            // calculate center point
            const double dem_y = y0 + dy * (0.5 + ii) / geogrid_upsampling;
            const double dem_x = x0 + dx * (0.5 + jj) / geogrid_upsampling;
            const Vec3 dem_c = {dem_x, dem_y,
                                dem_interp_block.interpolateXY(dem_x, dem_y)};
            double a_c = (a00 + a01 + a10 + a11) / 4.0;
            double r_c = (r00 + r01 + r10 + r11) / 4.0;

            converged = geo2rdr(proj->inverse(dem_c), ellipsoid, orbit, dop,
                                a_c, r_c, radar_grid.wavelength(), side,
                                threshold, num_iter, delta_range);

            if (!converged) {
                a_c = std::numeric_limits<double>::quiet_NaN();
                r_c = std::numeric_limits<double>::quiet_NaN();
            }
            if (out_geo_grid != nullptr) {
                out_geo_grid_a(i, jj) = (a_c - start) / pixazm;
                out_geo_grid_r(i, jj) = (r_c - r0) / dr;
            }
            if (!converged)
                continue;

            // Set DEM-coordinate corner vectors
            const Vec3 xyz00 = ellipsoid.lonLatToXyz(proj->inverse(dem00));
            const Vec3 xyz10 = ellipsoid.lonLatToXyz(proj->inverse(dem10));
            const Vec3 xyz01 = ellipsoid.lonLatToXyz(proj->inverse(dem01));
            const Vec3 xyz11 = ellipsoid.lonLatToXyz(proj->inverse(dem11));
            const Vec3 xyz_c = ellipsoid.lonLatToXyz(proj->inverse(dem_c));

            // Calculate look vector
            isce::core::cartesian_t xyz_plat, vel;
            isce::error::ErrorCode status = orbit.interpolate(
                    &xyz_plat, &vel, a_c, OrbitInterpBorderMode::FillNaN);
            if (status != isce::error::ErrorCode::Success)
                continue;

            const Vec3 lookXYZ = (xyz_plat - xyz_c).normalized();

            // Prepare call to computeFacet()
            double p00_c = (xyz00 - xyz_c).norm();
            double p10_c, p01_c, p11_c, divisor = 1;
            if (rtc_area_mode == rtcAreaMode::AREA_FACTOR)
                divisor = (radar_grid.rangePixelSpacing() * vel.norm() *
                           radar_grid.azimuthTimeInterval());

            if (input_radiometry ==
                rtcInputRadiometry::SIGMA_NAUGHT_ELLIPSOID) {
                const double costheta = xyz_c.dot(lookXYZ) / xyz_c.norm();

                // Compute incidence angle components
                const double sintheta = std::sqrt(1. - costheta * costheta);
                divisor /= sintheta;
            }

            bool clockwise_direction = (dem_interp_block.deltaY() > 0);

            // Prepare call to _addArea()
            int size_x = x_max - x_min + 1;
            int size_y = y_max - y_min + 1;
            isce::core::Matrix<double> w_arr_1(size_y, size_x);
            isce::core::Matrix<double> w_arr_2(size_y, size_x);
            w_arr_1.fill(0);
            w_arr_2.fill(0);

            double nlooks_1 = 0, nlooks_2 = 0;

            double y00 = (a00 - start) / pixazm - y_min;
            double y10 = (a10 - start) / pixazm - y_min;
            double y01 = (a01 - start) / pixazm - y_min;
            double y11 = (a11 - start) / pixazm - y_min;
            double y_c = (a_c - start) / pixazm - y_min;

            double x00 = (r00 - r0) / dr - x_min;
            double x10 = (r10 - r0) / dr - x_min;
            double x01 = (r01 - r0) / dr - x_min;
            double x11 = (r11 - r0) / dr - x_min;
            double x_c = (r_c - r0) / dr - x_min;

            int plane_orientation;
            if (radar_grid.lookSide() == isce::core::LookSide::Left)
                plane_orientation = -1;
            else
                plane_orientation = 1;

            areaProjIntegrateSegment(y_c, y00, x_c, x00, size_y, size_x,
                                     w_arr_1, nlooks_1, plane_orientation);

            // Compute the area (first facet)
            double area = computeFacet(xyz_c, xyz00, xyz01, lookXYZ, p00_c,
                                       p01_c, divisor, clockwise_direction);
            // Add area to output grid
            _addArea(area, out_array, radar_grid_nlooks, out_nlooks_array,
                     radar_grid.length(), radar_grid.width(), x_min, y_min,
                     size_x, size_y, w_arr_1, nlooks_1, w_arr_2, nlooks_2, x_c,
                     x00, x01, y_c, y00, y01, plane_orientation);

            // Compute the area (second facet)
            area = computeFacet(xyz_c, xyz01, xyz11, lookXYZ, p01_c, p11_c,
                                divisor, clockwise_direction);

            // Add area to output grid
            _addArea(area, out_array, radar_grid_nlooks, out_nlooks_array,
                     radar_grid.length(), radar_grid.width(), x_min, y_min,
                     size_x, size_y, w_arr_2, nlooks_2, w_arr_1, nlooks_1, x_c,
                     x01, x11, y_c, y01, y11, plane_orientation);

            // Compute the area (third facet)
            area = computeFacet(xyz_c, xyz11, xyz10, lookXYZ, p11_c, p10_c,
                                divisor, clockwise_direction);

            // Add area to output grid
            _addArea(area, out_array, radar_grid_nlooks, out_nlooks_array,
                     radar_grid.length(), radar_grid.width(), x_min, y_min,
                     size_x, size_y, w_arr_1, nlooks_1, w_arr_2, nlooks_2, x_c,
                     x11, x10, y_c, y11, y10, plane_orientation);

            // Compute the area (fourth facet)
            area = computeFacet(xyz_c, xyz10, xyz00, lookXYZ, p10_c, p00_c,
                                divisor, clockwise_direction);

            // Add area to output grid
            _addArea(area, out_array, radar_grid_nlooks, out_nlooks_array,
                     radar_grid.length(), radar_grid.width(), x_min, y_min,
                     size_x, size_y, w_arr_2, nlooks_2, w_arr_1, nlooks_1, x_c,
                     x10, x00, y_c, y10, y00, plane_orientation);
        }
    }

    if (out_geo_vertices != nullptr)
#pragma omp critical
    {
        out_geo_vertices->setBlock(out_geo_vertices_a.data(), 0,
                                   block * block_size_with_upsampling, jmax + 1,
                                   this_block_size_with_upsampling + 1, 1);
        out_geo_vertices->setBlock(out_geo_vertices_r.data(), 0,
                                   block * block_size_with_upsampling, jmax + 1,
                                   this_block_size_with_upsampling + 1, 2);
    }

    if (out_geo_grid != nullptr)
#pragma omp critical
    {
        out_geo_grid->setBlock(out_geo_grid_a.data(), 0,
                               block * block_size_with_upsampling, jmax,
                               this_block_size, 1);
        out_geo_grid->setBlock(out_geo_grid_r.data(), 0,
                               block * block_size_with_upsampling, jmax,
                               this_block_size, 2);
    }
}

void facetRTCAreaProj(
        isce::io::Raster& dem_raster, isce::io::Raster& output_raster,
        const isce::product::RadarGridParameters& radar_grid,
        const isce::core::Orbit& orbit,
        const isce::core::LUT2d<double>& input_dop, const double y0,
        const double dy, const double x0, const double dx,
        const int geogrid_length, const int geogrid_width, const int epsg,
        rtcInputRadiometry input_radiometry, rtcAreaMode rtc_area_mode,
        double geogrid_upsampling, float rtc_min_value_db,
        float radar_grid_nlooks,
        isce::io::Raster* out_geo_vertices, isce::io::Raster* out_geo_grid,
        isce::io::Raster* out_nlooks, rtcMemoryMode rtc_memory_mode,
        isce::core::dataInterpMethod interp_method, double threshold,
        int num_iter, double delta_range) {
    /*
      Description of the area projection algorithm can be found in Geocode.cpp
    */

    pyre::journal::info_t info("isce.geometry.facetRTCAreaProj");

    if (std::isnan(geogrid_upsampling))
        geogrid_upsampling = 2;

    assert(geogrid_length > 0);
    assert(geogrid_width > 0);
    assert(geogrid_upsampling > 0);

    // Ellipsoid being used for processing
    std::unique_ptr<isce::core::ProjectionBase> proj(
            isce::core::createProj(epsg));
    const isce::core::Ellipsoid& ellipsoid = proj->ellipsoid();

    print_parameters(info, radar_grid, y0, dy, x0, dx, geogrid_length,
                     geogrid_width, input_radiometry, rtc_area_mode,
                     geogrid_upsampling, rtc_min_value_db);

    // start (az) and r0 at the outer edge of the first pixel:
    const double pixazm = radar_grid.azimuthTimeInterval();
    double start = radar_grid.sensingStart() - 0.5 * pixazm;
    const double dr = radar_grid.rangePixelSpacing();
    double r0 = radar_grid.startingRange() - 0.5 * dr;

    // Bounds for valid RDC coordinates
    int xbound = radar_grid.width() - 1.0;
    int ybound = radar_grid.length() - 1.0;

    const int imax = geogrid_length * geogrid_upsampling;
    const int jmax = geogrid_width * geogrid_upsampling;

    // Output raster
    isce::core::Matrix<float> out_array(radar_grid.length(),
                                        radar_grid.width());
    out_array.fill(0);
    isce::core::Matrix<float> out_nlooks_array;
    if (out_nlooks != nullptr) {
        out_nlooks_array.resize(radar_grid.length(), radar_grid.width());
        out_nlooks_array.fill(0);
    }

    const int progress_block = imax * jmax / 100;
    int numdone = 0;
    int min_block_length = 32;
    int block_size, block_size_with_upsampling;

    int nblocks;

    if (rtc_memory_mode == rtcMemoryMode::RTC_SINGLE_BLOCK) {
        nblocks = 1;
        block_size_with_upsampling = imax;
        block_size = geogrid_length;
    } else {
        nblocks = areaProjGetNBlocks(imax, &info, geogrid_upsampling,
                                     &block_size_with_upsampling, &block_size,
                                     min_block_length);
    }

    info << "block size: " << block_size << pyre::journal::endl;
    info << "block size (with upsampling): " << block_size_with_upsampling
         << pyre::journal::endl;

#pragma omp parallel for schedule(dynamic)
    for (int block = 0; block < nblocks; ++block) {
        _RunBlock(jmax, block_size, block_size_with_upsampling, block, numdone,
                  progress_block, geogrid_upsampling, interp_method, dem_raster,
                  out_geo_vertices, out_geo_grid, start, pixazm, dr, r0, xbound,
                  ybound, y0, dy, x0, dx, geogrid_length, geogrid_width,
                  radar_grid, input_dop, ellipsoid, orbit, threshold, num_iter,
                  delta_range, out_array, out_nlooks_array, proj.get(),
                  rtc_area_mode, input_radiometry, radar_grid_nlooks);
    }

    printf("\rRTC progress: 100%%\n");
    std::cout << std::endl;

    if (!std::isnan(rtc_min_value_db) &&
        rtc_area_mode == rtcAreaMode::AREA_FACTOR) {
        float rtc_min_value = std::pow(10, (rtc_min_value_db / 10));
        info << "applying min. RTC value: " << rtc_min_value_db
             << " [dB] ~= " << rtc_min_value << pyre::journal::endl;
        for (int i = 0; i < radar_grid.length(); ++i)
            for (int j = 0; j < radar_grid.width(); ++j) {
                if (out_array(i, j) >= rtc_min_value)
                    continue;
                out_array(i, j) = std::numeric_limits<float>::quiet_NaN();
            }
    }

    output_raster.setBlock(out_array.data(), 0, 0, radar_grid.width(),
                           radar_grid.length());

    if (out_geo_vertices != nullptr) {
        double geotransform_edges[] = {x0 - dx / 2.0,
                                       dx / geogrid_upsampling,
                                       0,
                                       y0 - dy / 2.0,
                                       0,
                                       dy / geogrid_upsampling};
        out_geo_vertices->setGeoTransform(geotransform_edges);
        out_geo_vertices->setEPSG(epsg);
    }

    if (out_geo_grid != nullptr) {
        double geotransform_grid[] = {x0, dx / geogrid_upsampling, 0, y0,
                                      0,  dy / geogrid_upsampling};
        out_geo_grid->setGeoTransform(geotransform_grid);
        out_geo_grid->setEPSG(epsg);
    }

    if (out_nlooks != nullptr)
        out_nlooks->setBlock(out_nlooks_array.data(), 0, 0, radar_grid.width(),
                             radar_grid.length());
}

/** Convert enum input_radiometry to string */
std::string get_input_radiometry_str(rtcInputRadiometry input_radiometry) {
    std::string input_radiometry_str;
    switch (input_radiometry) {
    case rtcInputRadiometry::BETA_NAUGHT:
        input_radiometry_str = "beta-naught";
        break;
    case rtcInputRadiometry::SIGMA_NAUGHT_ELLIPSOID:
        input_radiometry_str = "sigma-naught";
        break;
    default:
        std::string error_message =
                "ERROR invalid radiometric terrain radiometry";
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), error_message);
    }
    return input_radiometry_str;
}

/** Convert enum output_mode to string */
std::string get_rtc_area_mode_str(rtcAreaMode rtc_area_mode) {
    std::string rtc_area_mode_str;
    switch (rtc_area_mode) {
    case rtcAreaMode::AREA: rtc_area_mode_str = "area"; break;
    case rtcAreaMode::AREA_FACTOR: rtc_area_mode_str = "area factor"; break;
    default:
        std::string error_message = "ERROR invalid RTC area mode";
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), error_message);
        break;
    }
    return rtc_area_mode_str;
}

/** Convert enum output_mode to string */
std::string get_rtc_algorithm_str(rtcAlgorithm rtc_algorithm) {
    std::string rtc_algorithm_str;
    switch (rtc_algorithm) {
    case rtcAlgorithm::RTC_DAVID_SMALL:
        rtc_algorithm_str = "David Small";
        break;
    case rtcAlgorithm::RTC_AREA_PROJECTION:
        rtc_algorithm_str = "Area projection";
        break;
    default:
        std::string error_message = "ERROR invalid RTC algorithm";
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), error_message);
        break;
    }
    return rtc_algorithm_str;
}

void print_parameters(pyre::journal::info_t& channel,
                      const isce::product::RadarGridParameters& radar_grid,
                      const double y0, const double dy, const double x0,
                      const double dx, const int geogrid_length,
                      const int geogrid_width,
                      rtcInputRadiometry input_radiometry,
                      rtcAreaMode rtc_area_mode, double geogrid_upsampling,
                      float rtc_min_value_db) {
    double yf = y0 + geogrid_length * dy;

    std::string input_radiometry_str =
            get_input_radiometry_str(input_radiometry);
    std::string rtc_area_mode_str = get_rtc_area_mode_str(rtc_area_mode);

    double ymax = std::max(y0, yf);
    double ymin = std::min(y0, yf);

    channel << "input radiometry: " << input_radiometry_str
            << pyre::journal::newline
            << "RTC area mode (area/area factor): " << rtc_area_mode_str
            << pyre::journal::newline
            << "Geogrid bounds:" << pyre::journal::newline << "Top Left: " << x0
            << " " << ymax << pyre::journal::newline
            << "Bottom Right: " << x0 + dx * (geogrid_width - 1) << " " << ymin
            << pyre::journal::newline << "Spacing: " << dx << " " << dy
            << pyre::journal::newline << "Dimensions: " << geogrid_width << " "
            << geogrid_length << pyre::journal::newline
            << "geogrid_upsampling: " << geogrid_upsampling
            << pyre::journal::newline << "look side: " << radar_grid.lookSide()
            << pyre::journal::newline
            << "radar_grid length: " << radar_grid.length()
            << ", width: " << radar_grid.width() << pyre::journal::newline
            << "RTC min value [dB]: " << rtc_min_value_db
            << pyre::journal::endl;
}
} // namespace geometry
} // namespace isce
