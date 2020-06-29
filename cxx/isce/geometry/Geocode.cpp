//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Gustavo H. X. Shiroma
// Copyright 2019-

#include "Geocode.h"

#include <algorithm>
#include <cmath>
#include <cpl_virtualmem.h>
#include <isce/core/Basis.h>
#include <isce/core/DenseMatrix.h>
#include <isce/core/Projections.h>
#include <isce/geometry/boundingbox.h>
#include <isce/geometry/geometry.h>
#include <isce/signal/Looks.h>
#include <limits>
#include <type_traits>

#include "DEMInterpolator.h"
#include "RTC.h"

using isce::core::OrbitInterpBorderMode;
using isce::core::Vec3;

namespace isce {
namespace geometry {

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

template<typename T, typename T_out> void _convertToOutputType(T a, T_out& b) {
    b = a;
}

template<typename T, typename T_out>
void _convertToOutputType(std::complex<T> a, T_out& b) {
    b = std::norm(a);
}

template<typename T, typename T_out>
void _convertToOutputType(std::complex<T> a, std::complex<T_out>& b) {
    b = a;
}

template<typename T, typename T_out>
void _accumulate(T_out& band_value, T a, double b) {
    if (b == 0)
        return;
    T_out a2;
    _convertToOutputType(a, a2);
    band_value += a2 * b;
}

template <typename T> struct is_complex_t : std::false_type {};
template <typename T> struct is_complex_t<std::complex<T>> : std::true_type {};
template <typename T>
constexpr bool is_complex() { return is_complex_t<T>::value; }

template<class T>
void Geocode<T>::updateGeoGrid(
        const isce::product::RadarGridParameters& radar_grid,
        isce::io::Raster& dem_raster) {

    pyre::journal::info_t info("isce.geometry.Geocode.updateGeoGrid");

    if (_epsgOut == 0)
        _epsgOut = dem_raster.getEPSG();

    if (std::isnan(_geoGridSpacingX))
        _geoGridSpacingX = dem_raster.dx();

    if (std::isnan(_geoGridSpacingY))
        _geoGridSpacingY = dem_raster.dy();

    if (std::isnan(_geoGridStartX) || std::isnan(_geoGridStartY) ||
        _geoGridLength <= 0 || _geoGridWidth <= 0) {
        std::unique_ptr<isce::core::ProjectionBase> proj(
                isce::core::createProj(_epsgOut));
        BoundingBox bbox = getGeoBoundingBoxHeightSearch(radar_grid, _orbit,
                                                         proj.get(), _doppler);
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
                         int width, int length, int epsgcode) {

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

template<class T>
void Geocode<T>::geocode(
        const isce::product::RadarGridParameters& radar_grid,
        isce::io::Raster& input_raster, isce::io::Raster& output_raster,
        isce::io::Raster& dem_raster, geocodeOutputMode output_mode,
        double geogrid_upsampling,
        isce::geometry::rtcInputRadiometry input_radiometry, int exponent,
        float rtc_min_value_db, 
        double rtc_geogrid_upsampling,
        rtcAlgorithm rtc_algorithm,
        double abs_cal_factor,
        float clip_min,
        float clip_max,
        float min_nlooks, 
        float radar_grid_nlooks,
        isce::io::Raster* out_geo_vertices,
        isce::io::Raster* out_dem_vertices,
        isce::io::Raster* out_geo_nlooks, isce::io::Raster* out_geo_rtc,
        isce::io::Raster* input_rtc, isce::io::Raster* output_rtc,
        geocodeMemoryMode geocode_memory_mode,
        isce::core::dataInterpMethod interp_method) {
    bool flag_complex_to_real = isce::signal::verifyComplexToRealCasting(
            input_raster, output_raster, exponent);

    if (output_mode == geocodeOutputMode::INTERP && !flag_complex_to_real)
        geocodeInterp<T>(radar_grid, input_raster, output_raster, dem_raster);
    else if (output_mode == geocodeOutputMode::INTERP &&
             (std::is_same<T, double>::value ||
              std::is_same<T, std::complex<double>>::value))
        geocodeInterp<double>(radar_grid, input_raster, output_raster,
                              dem_raster);
    else if (output_mode == geocodeOutputMode::INTERP)
        geocodeInterp<float>(radar_grid, input_raster, output_raster,
                             dem_raster);
    else if (!flag_complex_to_real)
        geocodeAreaProj<T>(radar_grid, input_raster, output_raster, dem_raster,
                           output_mode, geogrid_upsampling, input_radiometry,
                           rtc_min_value_db, rtc_geogrid_upsampling,
                           rtc_algorithm, abs_cal_factor, 
                           clip_min, clip_max, min_nlooks,
                           radar_grid_nlooks,
                           out_geo_vertices, out_dem_vertices, out_geo_nlooks,
                           out_geo_rtc, input_rtc, output_rtc,
                           geocode_memory_mode, interp_method);
    else if (std::is_same<T, double>::value ||
             std::is_same<T, std::complex<double>>::value)
        geocodeAreaProj<double>(
                radar_grid, input_raster, output_raster, dem_raster,
                output_mode, geogrid_upsampling, input_radiometry,
                rtc_min_value_db, rtc_geogrid_upsampling, rtc_algorithm,
                abs_cal_factor, 
                clip_min, clip_max, min_nlooks,
                radar_grid_nlooks, out_geo_vertices,
                out_dem_vertices, out_geo_nlooks, out_geo_rtc, input_rtc,
                output_rtc, geocode_memory_mode, interp_method);
    else
        geocodeAreaProj<float>(
                radar_grid, input_raster, output_raster, dem_raster,
                output_mode, geogrid_upsampling, input_radiometry,
                rtc_min_value_db, rtc_geogrid_upsampling, rtc_algorithm,
                abs_cal_factor, 
                clip_min, clip_max, min_nlooks,
                radar_grid_nlooks, out_geo_vertices,
                out_dem_vertices, out_geo_nlooks, out_geo_rtc, input_rtc,
                output_rtc, geocode_memory_mode, interp_method);
}

template<class T>
template<class T_out>
void Geocode<T>::geocodeInterp(
        const isce::product::RadarGridParameters& radar_grid,
        isce::io::Raster& inputRaster, isce::io::Raster& outputRaster,
        isce::io::Raster& demRaster) {

    std::unique_ptr<isce::core::Interpolator<T_out>> interp {
            isce::core::createInterpolator<T_out>(_interp_method)};

    // number of bands in the input raster
    int nbands = inputRaster.numBands();

    // create projection based on _epsg code
    std::unique_ptr<isce::core::ProjectionBase> proj(
            isce::core::createProj(_epsgOut));

    // instantiate the DEMInterpolator
    DEMInterpolator demInterp;

    // Compute number of blocks in the output geocoded grid
    int nBlocks = _geoGridLength / _linesPerBlock;
    if ((_geoGridLength % _linesPerBlock) != 0)
        nBlocks += 1;

    std::cout << "nBlocks: " << nBlocks << std::endl;
    // loop over the blocks of the geocoded Grid
    for (int block = 0; block < nBlocks; ++block) {
        std::cout << "block: " << block << std::endl;
        // Get block extents (of the geocoded grid)
        int lineStart, geoBlockLength;
        lineStart = block * _linesPerBlock;
        if (block == (nBlocks - 1)) {
            geoBlockLength = _geoGridLength - lineStart;
        } else {
            geoBlockLength = _linesPerBlock;
        }
        int blockSize = geoBlockLength * _geoGridWidth;

        // First and last line of the data block in radar coordinates
        size_t azimuthFirstLine = radar_grid.length() - 1;
        size_t azimuthLastLine = 0;

        // First and last pixel of the data block in radar coordinates
        size_t rangeFirstPixel = radar_grid.width() - 1;
        size_t rangeLastPixel = 0;

        // load a block of DEM for the current geocoded grid
        _loadDEM(demRaster, demInterp, proj.get(), lineStart, geoBlockLength,
                 _geoGridWidth, _demBlockMargin);

        // X and Y indices (in the radar coordinates) for the
        // geocoded pixels (after geo2rdr computation)
        std::valarray<double> radarX(blockSize);
        std::valarray<double> radarY(blockSize);

#pragma omp parallel shared(azimuthFirstLine, rangeFirstPixel,                 \
                            azimuthLastLine, rangeLastPixel)
        {
            // Init thread-local swath extents
            size_t localAzimuthFirstLine = radar_grid.length() - 1;
            size_t localAzimuthLastLine = 0;
            size_t localRangeFirstPixel = radar_grid.width() - 1;
            size_t localRangeLastPixel = 0;

// Loop over lines, samples of the output grid
#pragma omp for collapse(2)
            for (int blockLine = 0; blockLine < geoBlockLength; ++blockLine) {
                for (int pixel = 0; pixel < _geoGridWidth; ++pixel) {

                    // Global line index
                    const int line = lineStart + blockLine;

                    // y coordinate in the out put grid
                    double y = _geoGridStartY + _geoGridSpacingY * (0.5 + line);

                    // x in the output geocoded Grid
                    double x =
                            _geoGridStartX + _geoGridSpacingX * (0.5 + pixel);

                    // Consistency check

                    // compute the azimuth time and slant range for the
                    // x,y coordinates in the output grid
                    double aztime, srange;
                    _geo2rdr(radar_grid, x, y, aztime, srange, demInterp,
                             proj.get());

                    if (std::isnan(aztime) || std::isnan(srange))
                        continue;

                    // get the row and column index in the radar grid
                    double rdrX, rdrY;
                    rdrY = ((aztime - radar_grid.sensingStart()) /
                            radar_grid.azimuthTimeInterval());

                    rdrX = ((srange - radar_grid.startingRange()) /
                            radar_grid.rangePixelSpacing());

                    if (rdrY < 0 || rdrX < 0 || rdrY >= radar_grid.length() ||
                        rdrX >= radar_grid.width())
                        continue;

                    localAzimuthFirstLine = std::min(localAzimuthFirstLine,
                                                     (size_t) std::floor(rdrY));
                    localAzimuthLastLine = std::max(
                            localAzimuthLastLine, (size_t) std::ceil(rdrY) - 1);
                    localRangeFirstPixel = std::min(localRangeFirstPixel,
                                                    (size_t) std::floor(rdrX));
                    localRangeLastPixel = std::max(
                            localRangeLastPixel, (size_t) std::ceil(rdrX) - 1);

                    // store the adjusted X and Y indices
                    radarX[blockLine * _geoGridWidth + pixel] = rdrX;
                    radarY[blockLine * _geoGridWidth + pixel] = rdrY;

                } // end loop over pixels of output grid
            }     // end loops over lines of output grid

#pragma omp critical
            {
                // Get min and max swath extents from among all threads
                azimuthFirstLine = std::min(azimuthFirstLine, localAzimuthFirstLine);
                azimuthLastLine = std::max(azimuthLastLine, localAzimuthLastLine);
                rangeFirstPixel = std::min(rangeFirstPixel, localRangeFirstPixel);
                rangeLastPixel = std::max(rangeLastPixel, localRangeLastPixel);
            }
        }

        if (azimuthFirstLine > azimuthLastLine ||
            rangeFirstPixel > rangeLastPixel)
            continue;

        // shape of the required block of data in the radar coordinates
        int rdrBlockLength = azimuthLastLine - azimuthFirstLine + 1;
        int rdrBlockWidth = rangeLastPixel - rangeFirstPixel + 1;

        // define the matrix based on the rasterbands data type
        isce::core::Matrix<T_out> rdrDataBlock(rdrBlockLength, rdrBlockWidth);
        isce::core::Matrix<T_out> geoDataBlock(geoBlockLength, _geoGridWidth);

        // fill both matrices with NaN
        rdrDataBlock.fill(std::numeric_limits<T_out>::quiet_NaN());
        geoDataBlock.fill(std::numeric_limits<T_out>::quiet_NaN());

        //for each band in the input:
        for (int band = 0; band < nbands; ++band) {
            std::cout << "band: " << band << std::endl;
            // get a block of data
            std::cout << "get data block " << std::endl;
            if ((std::is_same<T, std::complex<float>>::value ||
                 std::is_same<T, std::complex<double>>::value)
                    &&(std::is_same<T_out, float>::value ||
                       std::is_same<T_out, double>::value) ) {
                isce::core::Matrix<T> rdrDataBlockTemp(rdrBlockLength,
                                                       rdrBlockWidth);
                inputRaster.getBlock(rdrDataBlockTemp.data(), rangeFirstPixel,
                                     azimuthFirstLine, rdrBlockWidth,
                                     rdrBlockLength, band + 1);
                for (int i = 0; i < rdrBlockLength; ++i)
                    for (int j = 0; j < rdrBlockWidth; ++j) {
                        T_out output_value;
                        _convertToOutputType(rdrDataBlockTemp(i, j),
                                             output_value);
                        rdrDataBlock(i, j) = output_value;
                    }
            } else
                inputRaster.getBlock(rdrDataBlock.data(), rangeFirstPixel,
                                     azimuthFirstLine, rdrBlockWidth,
                                     rdrBlockLength, band + 1);

            // interpolate the data in radar grid to the geocoded grid
            std::cout << "interpolate " << std::endl;
            _interpolate(rdrDataBlock, geoDataBlock, radarX, radarY,
                         rdrBlockWidth, rdrBlockLength, azimuthFirstLine,
                         rangeFirstPixel, interp.get());

            // set output
            std::cout << "set output " << std::endl;
            outputRaster.setBlock(geoDataBlock.data(), 0, lineStart,
                                  _geoGridWidth, geoBlockLength, band + 1);
        }
        // set output block of data
    } // end loop over block of output grid

    double geotransform[] = {
            _geoGridStartX,  _geoGridSpacingX, 0, _geoGridStartY, 0,
            _geoGridSpacingY};
    if (_geoGridSpacingY > 0) {
        geotransform[3] = _geoGridStartY + _geoGridLength * _geoGridSpacingY;
        geotransform[5] = -_geoGridSpacingY;
    }

    outputRaster.setGeoTransform(geotransform);

    outputRaster.setEPSG(_epsgOut);
}

template<class T>
template<class T_out>
void Geocode<T>::_interpolate(isce::core::Matrix<T_out>& rdrDataBlock,
                              isce::core::Matrix<T_out>& geoDataBlock,
                              std::valarray<double>& radarX,
                              std::valarray<double>& radarY,
                              int radarBlockWidth, int radarBlockLength,
                              int azimuthFirstLine, int rangeFirstPixel,
                              isce::core::Interpolator<T_out>* _interp) {
    auto length = geoDataBlock.length();
    auto width = geoDataBlock.width();
    double extraMargin = 4.0;

#pragma omp parallel for
    for (decltype(length) kk = 0; kk < length * width; ++kk) {

        auto i = kk / width;
        auto j = kk % width;

        // adjust the row and column indicies for the current block,
        // i.e., moving the origin to the top-left of this radar block.
        double rdrY = radarY[i * width + j] - azimuthFirstLine;
        double rdrX = radarX[i * width + j] - rangeFirstPixel;

        if (rdrX < extraMargin || rdrY < extraMargin ||
            rdrX >= (radarBlockWidth - extraMargin) ||
            rdrY >= (radarBlockLength - extraMargin))
            continue;
        geoDataBlock(i, j) = _interp->interpolate(rdrX, rdrY, rdrDataBlock);
    }
}

template<class T>
void Geocode<T>::_loadDEM(isce::io::Raster& demRaster,
                          DEMInterpolator& demInterp,
                          isce::core::ProjectionBase* proj, int lineStart,
                          int blockLength, int blockWidth, double demMargin) {
    // Create projection for DEM
    int epsgcode = demRaster.getEPSG();

    // Initialize bounds
    double minX = -1.0e64;
    double maxX = 1.0e64;
    double minY = -1.0e64;
    double maxY = 1.0e64;

    // Projection systems are different
    if (epsgcode != proj->code()) {

        // Create transformer to match the DEM
        std::unique_ptr<isce::core::ProjectionBase> demproj(
                isce::core::createProj(epsgcode));

        // Skip factors
        const int askip = std::max(static_cast<int>(blockLength / 10.), 1);
        const int rskip = std::max(static_cast<int>(blockWidth / 10.), 1);

        // Construct vectors of line/pixel indices to traverse perimeter
        std::vector<int> lineInd, pixInd;

        // Top edge
        for (int j = 0; j < blockWidth; j += rskip) {
            lineInd.emplace_back(0);
            pixInd.emplace_back(j);
        }

        // Right edge
        for (int i = 0; i < blockLength; i += askip) {
            lineInd.emplace_back(i);
            pixInd.emplace_back(blockWidth);
        }

        // Bottom edge
        for (int j = blockWidth; j > 0; j -= rskip) {
            lineInd.emplace_back(blockLength - 1);
            pixInd.emplace_back(j);
        }

        // Left edge
        for (int i = blockLength; i > 0; i -= askip) {
            lineInd.emplace_back(i);
            pixInd.emplace_back(0);
        }

        // Loop over the indices
        for (int i = 0; i < lineInd.size(); ++i) {
            Vec3 outpt = {_geoGridStartX + _geoGridSpacingX * (0.5 + pixInd[i]),
                          _geoGridStartY +
                                  _geoGridSpacingY * (0.5 + lineInd[i]),
                          0.0};

            Vec3 dempt;
            if (!projTransform(proj, demproj.get(), outpt, dempt)) {
                minX = std::min(minX, dempt[0]);
                maxX = std::max(maxX, dempt[0]);
                minY = std::min(minY, dempt[1]);
                maxY = std::max(maxY, dempt[1]);
            }
        }
    } else {
        // Use the corners directly as the projection system is the same
        double Y1 = _geoGridStartY + _geoGridSpacingY * lineStart;
        double Y2 =
                _geoGridStartY + _geoGridSpacingY * (lineStart + blockLength);
        minY = std::min(Y1, Y2);
        maxY = std::max(Y1, Y2);
        minX = _geoGridStartX;
        maxX = _geoGridStartX + _geoGridSpacingX * (blockWidth);
    }

    // If not LonLat, scale to meters
    demMargin = (epsgcode != 4326) ? isce::core::decimaldeg2meters(demMargin)
                                   : demMargin;

    // Account for margins
    minX -= demMargin;
    maxX += demMargin;
    minY -= demMargin;
    maxY += demMargin;

    // load the DEM for this bounding box
    demInterp.loadDEM(demRaster, minX, maxX, minY, maxY);

    if (demInterp.width() == 0 || demInterp.length() == 0)
        std::cout << "warning there are not enough DEM coverage in the "
                     "bounding box. "
                  << std::endl;

    // declare the dem interpolator
    demInterp.declare();
}

template<class T>
void Geocode<T>::_geo2rdr(const isce::product::RadarGridParameters& radar_grid,
                          double x, double y, double& azimuthTime,
                          double& slantRange, DEMInterpolator& demInterp,
                          isce::core::ProjectionBase* proj) {
    // coordinate in the output projection system
    const Vec3 xyz {x, y, 0.0};

    // transform the xyz in the output projection system to llh
    Vec3 llh = proj->inverse(xyz);

    // interpolate the height from the DEM for this pixel
    llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

    // Perform geo->rdr iterations
    int geostat = geo2rdr(llh, _ellipsoid, _orbit, _doppler, azimuthTime,
                          slantRange, radar_grid.wavelength(),
                          radar_grid.lookSide(), _threshold, _numiter, 1.0e-8);

    // Check convergence
    if (geostat == 0) {
        azimuthTime = std::numeric_limits<double>::quiet_NaN();
        slantRange = std::numeric_limits<double>::quiet_NaN();
        return;
    }
}

template<class T>
template<class T_out>
void Geocode<T>::geocodeAreaProj(
        const isce::product::RadarGridParameters& radar_grid,
        isce::io::Raster& input_raster, isce::io::Raster& output_raster,
        isce::io::Raster& dem_raster, geocodeOutputMode output_mode,
        double geogrid_upsampling,
        isce::geometry::rtcInputRadiometry input_radiometry,
        float rtc_min_value_db, 
        double rtc_geogrid_upsampling,
        rtcAlgorithm rtc_algorithm,
        double abs_cal_factor,
        float clip_min,
        float clip_max,
        float min_nlooks,
        float radar_grid_nlooks,
        isce::io::Raster * out_geo_vertices, 
        isce::io::Raster * out_dem_vertices,
        isce::io::Raster* out_geo_nlooks, isce::io::Raster* out_geo_rtc,
        isce::io::Raster* input_rtc, isce::io::Raster* output_rtc,
        geocodeMemoryMode geocode_memory_mode,
        isce::core::dataInterpMethod interp_method) {

    pyre::journal::info_t info("isce.geometry.Geocode.geocodeAreaProj");

    if (std::isnan(geogrid_upsampling))
        geogrid_upsampling = 1;
    assert(geogrid_upsampling > 0);
    assert(output_mode != geocodeOutputMode::INTERP);

    if (output_mode == geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT) {
        std::string input_radiometry_str =
                get_input_radiometry_str(input_radiometry);
        info << "input radiometry: " << input_radiometry_str
             << pyre::journal::endl;
    }

    if (!std::isnan(clip_min))
        info << "clip min: " << clip_min << pyre::journal::endl; 

    if (!std::isnan(clip_max))
        info << "clip max: " << clip_max << pyre::journal::endl;

    if (!std::isnan(min_nlooks))
        info << "nlooks min: " << min_nlooks << pyre::journal::endl;     

    isce::core::Matrix<float> rtc_area;
    if (output_mode == geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT) {

        // declare pointer to the raster containing the RTC area factor
        isce::io::Raster* rtc_raster;
        std::unique_ptr<isce::io::Raster> rtc_raster_unique_ptr;

        if (input_rtc == nullptr) {

            info << "calling RTC (from geocode)..." << pyre::journal::endl;

            // if RTC (area factor) raster does not needed to be saved,
            // initialize it as a GDAL memory virtual file
            if (output_rtc == nullptr) {
                rtc_raster_unique_ptr = std::make_unique<isce::io::Raster>(
                        "/vsimem/dummy", radar_grid.width(),
                        radar_grid.length(), 1, GDT_Float32, "ENVI");
                rtc_raster = rtc_raster_unique_ptr.get();
            }

            // Otherwise, copies the pointer to the output RTC file
            else
                rtc_raster = output_rtc;

            isce::geometry::rtcAreaMode rtc_area_mode =
                    isce::geometry::rtcAreaMode::AREA_FACTOR;

            if (std::isnan(rtc_geogrid_upsampling))
                rtc_geogrid_upsampling = 2 * geogrid_upsampling;

            isce::geometry::rtcMemoryMode rtc_memory_mode;
            if (geocode_memory_mode == geocodeMemoryMode::AUTO)
                rtc_memory_mode = isce::geometry::RTC_AUTO;
            else if (geocode_memory_mode == geocodeMemoryMode::SINGLE_BLOCK)
                rtc_memory_mode = isce::geometry::RTC_SINGLE_BLOCK;
            else
                rtc_memory_mode = isce::geometry::RTC_BLOCKS_GEOGRID;

            facetRTC(dem_raster, *rtc_raster, radar_grid, _orbit, _doppler,
                     _geoGridStartY, _geoGridSpacingY, _geoGridStartX,
                     _geoGridSpacingX, _geoGridLength, _geoGridWidth, _epsgOut,
                     input_radiometry, rtc_area_mode, rtc_algorithm,
                     rtc_geogrid_upsampling, rtc_min_value_db, radar_grid_nlooks,
                     nullptr, nullptr, nullptr, rtc_memory_mode,
                     interp_method, _threshold, _numiter, 1.0e-8);
        } else {
            info << "reading pre-computed RTC..." << pyre::journal::endl;
            rtc_raster = input_rtc;
        }

        rtc_area.resize(radar_grid.length(), radar_grid.width());
        rtc_raster->getBlock(rtc_area.data(), 0, 0, radar_grid.width(),
                             radar_grid.length(), 1);
    }

    // number of bands in the input raster
    int nbands = input_raster.numBands();

    info << "nbands: " << nbands << pyre::journal::endl;

    // create projection based on epsg code
    std::unique_ptr<isce::core::ProjectionBase> proj(
            isce::core::createProj(_epsgOut));

    const double pixazm = radar_grid.azimuthTimeInterval();
    const double start = radar_grid.sensingStart() - 0.5 * pixazm;

    const double dr = radar_grid.rangePixelSpacing();
    double r0 = radar_grid.startingRange() - 0.5 * dr;

    // Bounds for valid RDC coordinates
    int xbound = radar_grid.width() - 1;
    int ybound = radar_grid.length() - 1;

    const int imax = _geoGridLength * geogrid_upsampling;
    const int jmax = _geoGridWidth * geogrid_upsampling;

    info << "radar grid width: " << radar_grid.width()
         << ", length: " << radar_grid.length() << pyre::journal::endl;

    int epsgcode = dem_raster.getEPSG();

    if (epsgcode < 0) {
        std::string error_msg = "invalid DEM EPSG";
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    const int progress_block = imax * jmax / 100;

    double rtc_min_value = 0;

    if (!std::isnan(rtc_min_value_db) &&
        output_mode == geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT) {
        rtc_min_value = std::pow(10, (rtc_min_value_db / 10));
        info << "RTC min. value: " << rtc_min_value_db
             << " [dB] = " << rtc_min_value << pyre::journal::endl;
    }

    if (abs_cal_factor != 1)
        info << "absolute calibration factor: " << abs_cal_factor
            << pyre::journal::endl;

    bool is_radar_grid_single_block =
            (geocode_memory_mode != geocodeMemoryMode::BLOCKS_GEOGRID_AND_RADARGRID);

    std::vector<std::unique_ptr<isce::core::Matrix<T_out>>> rdrData;

    if (is_radar_grid_single_block) {
        rdrData.reserve(nbands);
        if (!std::is_same<T, T_out>::value) {

            info << "converting band to output dtype..." << pyre::journal::endl;
            int radargrid_nblocks, radar_block_size;

            if (geocode_memory_mode == geocodeMemoryMode::SINGLE_BLOCK) {
                radargrid_nblocks = 1;
                radar_block_size = radar_grid.length();
            } else {

                radargrid_nblocks = areaProjGetNBlocks(
                        radar_grid.length(), &info, 0, &radar_block_size);
            }

            for (int band = 0; band < nbands; ++band) {
                info << "reading input raster band: " << band + 1
                     << pyre::journal::endl;
                rdrData.emplace_back(std::make_unique<isce::core::Matrix<T_out>>(
                        radar_grid.length(), radar_grid.width()));
                #pragma omp parallel for schedule(dynamic)
                    for (int block = 0; block < radargrid_nblocks; ++block) {
                        int this_block_size = radar_block_size;
                        if ((block + 1) * radar_block_size > radar_grid.length())
                            this_block_size =
                                    radar_grid.length() % radar_block_size;
                        isce::core::Matrix<T> radar_data_out(
                                this_block_size, radar_grid.width());
                        #pragma omp critical
                        {
                            input_raster.getBlock(radar_data_out.data(), 0,
                                                  block * radar_block_size,
                                                  radar_grid.width(),
                                                  this_block_size, band + 1);
                        }
                        for (int i = 0; i < this_block_size; ++i) {
                            // initiating lower right vertex
                            int ii = block * radar_block_size + i;
                            for (int j = 0; j < radar_grid.width(); ++j) {
                                T_out radar_data_value;
                                _convertToOutputType(radar_data_out(i, j),
                                                     radar_data_value);
                                rdrData[band].get()->operator()(ii, j) =
                                        radar_data_value;
                            }
                        }
                    }
            }
        } else {
            for (int band = 0; band < nbands; ++band) {
                info << "reading input raster band: " << band + 1
                     << pyre::journal::endl;
                rdrData.emplace_back(
                        std::make_unique<isce::core::Matrix<T_out>>(
                                radar_grid.length(), radar_grid.width()));
                input_raster.getBlock(rdrData[band].get()->data(), 0, 0,
                                      radar_grid.width(), radar_grid.length(),
                                      band + 1);
            }
        }
    }

    int block_size, nblocks, block_size_with_upsampling;
    if (geocode_memory_mode == geocodeMemoryMode::SINGLE_BLOCK) {
        nblocks = 1;
        block_size = _geoGridLength;
        block_size_with_upsampling = imax;
    }
    else {
        if (geocode_memory_mode == 
            geocodeMemoryMode::BLOCKS_GEOGRID_AND_RADARGRID) {
            nblocks = areaProjGetNBlocks(imax, &info, geogrid_upsampling,
                                         &block_size_with_upsampling,
                                         &block_size);
        }
        else {
            int min_block_length = 32;
            nblocks = areaProjGetNBlocks(imax, &info, geogrid_upsampling,
                                         &block_size_with_upsampling,
                                         &block_size,
                                         min_block_length);
        }
    }

    int numdone = 0;

    info << "starting geocoding" << pyre::journal::endl; 

    #pragma omp parallel for schedule(dynamic)
    for (int block = 0; block < nblocks; ++block) {
        _RunBlock<T_out>(radar_grid, is_radar_grid_single_block, rdrData, jmax,
                         block_size, block_size_with_upsampling, block, numdone,
                         progress_block, geogrid_upsampling, nbands,
                         interp_method, dem_raster, 
                         out_geo_vertices,
                         out_dem_vertices,
                         out_geo_nlooks, out_geo_rtc, start,
                         pixazm, dr, r0, xbound, ybound, proj.get(), rtc_area,
                         input_raster, output_raster, output_mode,
                         rtc_min_value, abs_cal_factor,
                         clip_min, clip_max, min_nlooks,
                         radar_grid_nlooks, info);
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

    if (out_geo_vertices != nullptr) {
        double geotransform_edges[] = {_geoGridStartX - _geoGridSpacingX / 2.0,
                                       _geoGridSpacingX / geogrid_upsampling,
                                       0,
                                       _geoGridStartY - _geoGridSpacingY / 2.0,
                                       0,
                                       _geoGridSpacingY / geogrid_upsampling};
        out_geo_vertices->setGeoTransform(geotransform_edges);
        out_geo_vertices->setEPSG(_epsgOut);
    }

    if (out_dem_vertices != nullptr) {
        double geotransform_edges[] = {_geoGridStartX - _geoGridSpacingX / 2.0,
                                       _geoGridSpacingX / geogrid_upsampling,
                                       0,
                                       _geoGridStartY - _geoGridSpacingY / 2.0,
                                       0,
                                       _geoGridSpacingY / geogrid_upsampling};
        out_dem_vertices->setGeoTransform(geotransform_edges);
        out_dem_vertices->setEPSG(_epsgOut);
    }

    if (out_geo_nlooks != nullptr) {
        out_geo_nlooks->setGeoTransform(geotransform);
        out_geo_nlooks->setEPSG(_epsgOut);
    }

    if (out_geo_rtc != nullptr) {
        out_geo_rtc->setGeoTransform(geotransform);
        out_geo_rtc->setEPSG(_epsgOut);
    }
}

template<class T>
void Geocode<T>::_GetRadarPositionVect(
        double dem_pos_1, const int k_start, const int k_end,
        double geogrid_upsampling, double& a11, double& r11, double& a_min,
        double& r_min, double& a_max, double& r_max,
        std::vector<double>& a_vect, std::vector<double>& r_vect,
        std::vector<Vec3>& dem_vect,
        const isce::product::RadarGridParameters& radar_grid,
        isce::core::ProjectionBase* proj, DEMInterpolator& dem_interp_block,
        bool flag_direction_line) {

    for (int kk = k_start; kk <= k_end; ++kk) {
        const int k = kk - k_start;

        Vec3 dem11;
        if (flag_direction_line) {
            // flag_direction_line == true: y fixed, varies x
            const double dem_pos_2 =
                    _geoGridStartX + _geoGridSpacingX * kk / geogrid_upsampling;
            // {x, y, z}
            dem11 = {dem_pos_2, dem_pos_1,
                     dem_interp_block.interpolateXY(dem_pos_2, dem_pos_1)};
        } else {
            // flag_direction_line == false: x fixed, varies y
            const double dem_pos_2 =
                    _geoGridStartY + _geoGridSpacingY * kk / geogrid_upsampling;
            // {x, y, z}
            dem11 = {dem_pos_1, dem_pos_2,
                     dem_interp_block.interpolateXY(dem_pos_1, dem_pos_2)};
        }
        int converged =
                geo2rdr(proj->inverse(dem11), _ellipsoid, _orbit, _doppler, a11,
                        r11, radar_grid.wavelength(), radar_grid.lookSide(),
                        _threshold, _numiter, 1.0e-8);
        // if it didn't converge, set NaN
        if (!converged) {
            a11 = radar_grid.sensingMid();
            r11 = radar_grid.midRange();
            continue;
        }

        if (a11 < a_min)
            a_min = a11;
        if (a11 > a_max)
            a_max = a11;
        if (r11 < r_min)
            r_min = r11;
        if (r11 > r_max)
            r_max = r11;

        // otherwise, save solution
        a_vect[k] = a11;
        r_vect[k] = r11;
        dem_vect[k] = dem11;
    }
}

template<class T> std::string Geocode<T>::_get_nbytes_str(long nbytes) {
    std::string nbytes_str;
    if (nbytes < std::pow(2, 10))
        nbytes_str = std::to_string(nbytes) + "B";
    else if (nbytes < std::pow(2, 20))
        nbytes_str = std::to_string((int) std::ceil(nbytes / std::pow(2, 10))) +
                     "KB";
    else if (nbytes < std::pow(2, 30))
        nbytes_str = std::to_string((int) std::ceil(nbytes / std::pow(2, 20))) +
                     "MB";
    else
        nbytes_str = std::to_string((int) std::ceil(nbytes / std::pow(2, 30))) +
                     "GB";
    return nbytes_str;
}

template<class T>
template<class T_out>
void Geocode<T>::_RunBlock(
        const isce::product::RadarGridParameters& radar_grid,
        bool is_radar_grid_single_block,
        std::vector<std::unique_ptr<isce::core::Matrix<T_out>>>& rdrData,
        const int jmax, int block_size, int block_size_with_upsampling,
        int block, int& numdone, int progress_block, double geogrid_upsampling,
        int nbands, isce::core::dataInterpMethod interp_method,
        isce::io::Raster& dem_raster, isce::io::Raster* out_geo_vertices,
        isce::io::Raster* out_dem_vertices, isce::io::Raster* out_geo_nlooks,
        isce::io::Raster* out_geo_rtc, const double start, const double pixazm,
        const double dr, double r0, int xbound, int ybound,
        isce::core::ProjectionBase* proj, isce::core::Matrix<float>& rtc_area,
        isce::io::Raster& input_raster, isce::io::Raster& output_raster,
        isce::geometry::geocodeOutputMode output_mode,
        float rtc_min_value, double abs_cal_factor, float clip_min,
        float clip_max, float min_nlooks, float radar_grid_nlooks,
        pyre::journal::info_t& info)
{

    double abs_cal_factor_effective; 
    if (!is_complex_t<T_out>())
        abs_cal_factor_effective = abs_cal_factor;
    else
        abs_cal_factor_effective = std::sqrt(abs_cal_factor);

    int this_block_size = block_size;
    if ((block + 1) * block_size > _geoGridLength)
        this_block_size = _geoGridLength % block_size;

    const int this_block_size_with_upsampling =
            this_block_size * geogrid_upsampling;

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

    isce::core::Matrix<float> out_dem_vertices_array;
    if (out_dem_vertices != nullptr) {
        out_dem_vertices_array.resize(this_block_size_with_upsampling + 1,
                                  jmax + 1);
        out_dem_vertices_array.fill(std::numeric_limits<float>::quiet_NaN());
    }

    isce::core::Matrix<float> out_geo_nlooks_array;
    if (out_geo_nlooks != nullptr) {
        out_geo_nlooks_array.resize(this_block_size, _geoGridWidth);
        out_geo_nlooks_array.fill(std::numeric_limits<float>::quiet_NaN());
    }

    isce::core::Matrix<float> out_geo_rtc_array;
    if (out_geo_rtc != nullptr) {
        out_geo_rtc_array.resize(this_block_size, _geoGridWidth);
        out_geo_rtc_array.fill(std::numeric_limits<float>::quiet_NaN());
    }

    int ii_0 = block * block_size_with_upsampling;
    DEMInterpolator dem_interp_block(0, interp_method);

    const double minX = _geoGridStartX;
    const double maxX = _geoGridStartX + _geoGridSpacingX * _geoGridWidth;
    double minY = _geoGridStartY +
                  (((double) ii_0) / geogrid_upsampling * _geoGridSpacingY);
    double maxY = _geoGridStartY +
                  std::min(((double) ii_0) / geogrid_upsampling + block_size,
                           (double) _geoGridLength) *
                          _geoGridSpacingY;

    std::vector<std::unique_ptr<isce::core::Matrix<T_out>>> geoDataBlock;
    geoDataBlock.reserve(nbands);
    for (int band = 0; band < nbands; ++band)
        geoDataBlock.emplace_back(
                std::make_unique<isce::core::Matrix<T_out>>(block_size, jmax));

    for (int band = 0; band < nbands; ++band)
        geoDataBlock[band].get()->fill(0);

    const double margin_x = std::abs(_geoGridSpacingX) * 10;
    const double margin_y = std::abs(_geoGridSpacingY) * 10;

#pragma omp critical
    {
        dem_interp_block.loadDEM(dem_raster, minX - margin_x, maxX + margin_x,
                                 std::min(minY, maxY) - margin_y,
                                 std::max(minY, maxY) + margin_y);
    }

    /*
    Example:
    jmax = 7 (columns)
    geogrid_upsampling = 1
    this_block_size = this_block_size_with_upsampling = 4 rows

    - r_last: points to the upper vertices of last processed row (it starts with
              the first row) and it has jmax+1 elements:

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
     n_elements = i_end - i_star = (n_rows + 1) - 2 = n_rows - 1

     since we are working inside the block and with upsampling:
     n_elements = this_block_size_with_upsampling - 1
    */

    double a11 = radar_grid.sensingMid();
    double r11 = radar_grid.midRange();
    Vec3 dem11;

    double a_min = radar_grid.sensingMid();
    double r_min = radar_grid.midRange();
    double a_max = radar_grid.sensingMid();
    double r_max = radar_grid.midRange();

    // pre-compute radar positions on the top of the geogrid
    bool flag_direction_line = true;
    const int j_start = 0;
    double dem_y1 =
            _geoGridStartY + (_geoGridSpacingY * ii_0) / geogrid_upsampling;
    std::vector<double> a_last(jmax + 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_last(jmax + 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_last(jmax + 1,
                               {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()});
    _GetRadarPositionVect(dem_y1, j_start, jmax, geogrid_upsampling, a11, r11,
                          a_min, r_min, a_max, r_max, a_last, r_last, dem_last,
                          radar_grid, proj, dem_interp_block,
                          flag_direction_line);

    // pre-compute radar positions on the bottom of the geogrid
    dem_y1 = _geoGridStartY +
             ((_geoGridSpacingY * (ii_0 + this_block_size_with_upsampling)) /
              geogrid_upsampling);

    std::vector<double> a_bottom(jmax + 1,
                                 std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_bottom(jmax + 1,
                                 std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_bottom(jmax + 1,
                                 {std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()});
    _GetRadarPositionVect(dem_y1, j_start, jmax, geogrid_upsampling, a11, r11,
                          a_min, r_min, a_max, r_max, a_bottom, r_bottom,
                          dem_bottom, radar_grid, proj, dem_interp_block,
                          flag_direction_line);

    // pre-compute radar positions on the left side of the geogrid
    flag_direction_line = false;
    std::vector<double> a_left(this_block_size_with_upsampling - 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_left(this_block_size_with_upsampling - 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_left(this_block_size_with_upsampling - 1,
                               {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()});

    int i_start = (ii_0 + 1);
    int i_end = ii_0 + this_block_size_with_upsampling - 1;
    double dem_x1 = _geoGridStartX;

    _GetRadarPositionVect(dem_x1, i_start, i_end, geogrid_upsampling, a11, r11,
                          a_min, r_min, a_max, r_max, a_left, r_left, dem_left,
                          radar_grid, proj, dem_interp_block,
                          flag_direction_line);

    // pre-compute radar positions on the right side of the geogrid
    std::vector<double> a_right(this_block_size_with_upsampling - 1,
                                std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_right(this_block_size_with_upsampling - 1,
                                std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_right(this_block_size_with_upsampling - 1,
                                {std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN()});
    dem_x1 = _geoGridStartX + _geoGridSpacingX * jmax / geogrid_upsampling;
    _GetRadarPositionVect(dem_x1, i_start, i_end, geogrid_upsampling, a11, r11,
                          a_min, r_min, a_max, r_max, a_right, r_right,
                          dem_right, radar_grid, proj, dem_interp_block,
                          flag_direction_line);

    // load radar grid data
    int offset_x = 0, offset_y = 0;
    std::vector<std::unique_ptr<isce::core::Matrix<T_out>>> rdrDataBlock;
    if (!is_radar_grid_single_block) {
        int margin_pixels = 10;
        offset_y = std::max((int) std::floor((a_min - start) / pixazm) -
                                    margin_pixels,
                            (int) 0);
        offset_x = std::max((int) std::floor((r_min - r0) / dr) - margin_pixels,
                            (int) 0);
        int grid_size_y = std::min((int) std::ceil((a_max - start) / pixazm) +
                                           margin_pixels,
                                   (int) radar_grid.length()) -
                          offset_y;
        int grid_size_x =
                std::min((int) std::ceil((r_max - r0) / dr) + margin_pixels,
                         (int) radar_grid.width()) -
                offset_x;
        isce::product::RadarGridParameters radar_grid_block =
                radar_grid.offsetAndResize(offset_y, offset_x, grid_size_y,
                                           grid_size_x);

        long nbytes = (radar_grid_block.length() * radar_grid_block.width() *
                       nbands * sizeof(T));

        std::string nbytes_str = _get_nbytes_str(nbytes);
        std::string bands_str = "";
        if (nbands > 1)
            bands_str = " (" + std::to_string(nbands) + " bands)";

        info << "block " << block
             << " radar grid width: " << radar_grid_block.width()
             << ", length: " << radar_grid_block.length()
             << ", req. memory: " << nbytes_str << bands_str
             << pyre::journal::endl;

        rdrDataBlock.reserve(nbands);
        for (int band = 0; band < nbands; ++band) {
            rdrDataBlock.emplace_back(std::make_unique<isce::core::Matrix<T_out>>(
                    radar_grid_block.length(), radar_grid_block.width()));

            if (!std::is_same<T, T_out>::value) {
                info << "converting band to output dtype..." << pyre::journal::endl;
                isce::core::Matrix<T> radar_data_out( 
                    radar_grid_block.length(), radar_grid_block.width());
                #pragma omp critical
                input_raster.getBlock(radar_data_out.data(), offset_x,
                                      offset_y, radar_grid_block.width(),
                                      radar_grid_block.length(), band + 1);
                for (int i = 0; i < radar_grid_block.length(); ++i)
                    for (int j = 0; j < radar_grid_block.width(); ++j) {
                        T_out radar_data_value;
                        _convertToOutputType(radar_data_out(i, j),
                                             radar_data_value);
                        rdrDataBlock[band].get()->operator()(i, j) =
                                radar_data_value;
                    }
            } else {
                #pragma omp critical
                input_raster.getBlock(rdrDataBlock[band].get()->data(),
                                      offset_x, offset_y,
                                      radar_grid_block.width(),
                                      radar_grid_block.length(), band + 1);
            }
        }
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
    for (int i = 0; i < this_block_size_with_upsampling; ++i) {

        // initiating lower right vertex
        int ii = block * block_size_with_upsampling + i;
        a11 = a_left[i];
        r11 = r_left[i];
        dem11 = dem_left[i];

        // initiating lower edge geogrid lat/northing position
        dem_y1 = _geoGridStartY +
                 _geoGridSpacingY * (1.0 + ii) / geogrid_upsampling;

        for (int jj = 0; jj < (int) jmax; ++jj) {

#pragma omp atomic
            numdone++;
            if (numdone % progress_block == 0)
#pragma omp critical
                printf("\rgeocode progress: %d%%",
                       (int) numdone / progress_block),
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

            // update "last" arrays (from lower left vertex)
            if (!std::isnan(a10)) {
                a_last[jj] = a10;
                r_last[jj] = r10;
                dem_last[jj] = dem10;
            }

            int converged;
            if (i < this_block_size_with_upsampling - 1 && jj < jmax - 1) {
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
                dem11 = {dem_x1, dem_y1,
                         dem_interp_block.interpolateXY(dem_x1, dem_y1)};
                converged = geo2rdr(proj->inverse(dem11), _ellipsoid, _orbit,
                                    _doppler, a11, r11, radar_grid.wavelength(),
                                    radar_grid.lookSide(), _threshold, _numiter,
                                    1.0e-8);
                if (!converged) {
                    a11 = std::numeric_limits<double>::quiet_NaN();
                    r11 = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
            } else if (i >= this_block_size_with_upsampling - 1 &&
                       !std::isnan(a_bottom[jj + 1]) && !std::isnan(r_bottom[jj + 1])) {
                a11 = a_bottom[jj + 1];
                r11 = r_bottom[jj + 1];
                dem11 = dem_bottom[jj + 1];
            } else if (jj >= jmax - 1 && !std::isnan(a_right[i]) &&
                       !std::isnan(r_right[i])) {
                a11 = a_right[i];
                r11 = r_right[i];
                dem11 = dem_right[i];

            } else {
                a11 = std::numeric_limits<double>::quiet_NaN();
                r11 = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            // if last column, also update top-right "last" arrays (from lower
            //   right vertex)
            if (jj == jmax - 1) {
                a_last[jj+1] = a11;
                r_last[jj+1] = r11;
                dem_last[jj+1] = dem11;
            }

            // define slant-range window
            const int y_min = std::floor((std::min(std::min(a00, a01),
                                                   std::min(a10, a11)) -
                                          start) /
                                         pixazm) -
                              1;
            if (y_min < -isce::core::AREA_PROJECTION_RADAR_GRID_MARGIN || 
                    y_min > ybound + 1)
                continue;
            const int x_min = std::floor((std::min(std::min(r00, r01),
                                                   std::min(r10, r11)) -
                                          r0) /
                                         dr) -
                              1;
            if (x_min < -isce::core::AREA_PROJECTION_RADAR_GRID_MARGIN || 
                    x_min > xbound + 1)
                continue;
            const int y_max = std::ceil((std::max(std::max(a00, a01),
                                                  std::max(a10, a11)) -
                                         start) /
                                        pixazm) +
                              1;
            if (y_max > ybound + 1 + 
                isce::core::AREA_PROJECTION_RADAR_GRID_MARGIN || 
                    y_max < -1 || y_max < y_min)
                continue;
            const int x_max = std::ceil((std::max(std::max(r00, r01),
                                                  std::max(r10, r11)) -
                                         r0) /
                                        dr) +
                              1;
            if (x_max > xbound + 1 + 
                isce::core::AREA_PROJECTION_RADAR_GRID_MARGIN || 
                    x_max < -1 || x_max < x_min)
                continue;

            const double y00 = (a00 - start) / pixazm - y_min;
            const double y10 = (a10 - start) / pixazm - y_min;
            const double y01 = (a01 - start) / pixazm - y_min;
            const double y11 = (a11 - start) / pixazm - y_min;

            const double x00 = (r00 - r0) / dr - x_min;
            const double x10 = (r10 - r0) / dr - x_min;
            const double x01 = (r01 - r0) / dr - x_min;
            const double x11 = (r11 - r0) / dr - x_min;

            const int size_x = x_max - x_min + 1;
            const int size_y = y_max - y_min + 1;

            isce::core::Matrix<double> w_arr(size_y, size_x);
            w_arr.fill(0);
            double w_total = 0;
            int plane_orientation;
            if (radar_grid.lookSide() == isce::core::LookSide::Left)
                plane_orientation = -1;
            else
                plane_orientation = 1;

            areaProjIntegrateSegment(y00, y01, x00, x01, size_y, size_x, w_arr,
                                     w_total, plane_orientation);
            areaProjIntegrateSegment(y01, y11, x01, x11, size_y, size_x, w_arr,
                                     w_total, plane_orientation);
            areaProjIntegrateSegment(y11, y10, x11, x10, size_y, size_x, w_arr,
                                     w_total, plane_orientation);
            areaProjIntegrateSegment(y10, y00, x10, x00, size_y, size_x, w_arr,
                                     w_total, plane_orientation);

            double nlooks = 0;

            std::vector<T_out> cumulative_sum(nbands, 0);
            float area_total = 0;

            // add all slant-range elements that contributes to the geogrid
            // pixel
            for (int yy = 0; yy < size_y; ++yy) {
                for (int xx = 0; xx < size_x; ++xx) {
                    double w = w_arr(yy, xx);
                    int y = yy + y_min;
                    int x = xx + x_min;
                    if (w == 0 || w * w_total < 0)
                        continue;
                    else if (y < 0 || x < 0 || y >= radar_grid.length() ||
                             x >= radar_grid.width()) {
                        nlooks = std::numeric_limits<double>::quiet_NaN();
                        break;
                    }
                    w = std::abs(w);
                    if (output_mode == geocodeOutputMode::
                                               AREA_PROJECTION_GAMMA_NAUGHT) {
                        float rtc_value = rtc_area(y, x);
                        if (std::isnan(rtc_value) || rtc_value < rtc_min_value)
                            continue;
                        nlooks += w;
                        if (is_complex_t<T_out>())
                            rtc_value = std::sqrt(rtc_value);
                        area_total += rtc_value * w;
                        if (output_mode ==
                            geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT)
                            w /= rtc_value;
                    } else {
                        nlooks += w;
                    }

                    for (int band = 0; band < nbands; ++band) {
                        if (is_radar_grid_single_block) {
                            _accumulate(cumulative_sum[band],
                                         rdrData[band].get()->operator()(
                                                 y - offset_y, x - offset_x),
                                         w);
                        } else {
                            _accumulate(cumulative_sum[band],
                                       rdrDataBlock[band].get()->operator()(
                                               y - offset_y, x - offset_x),
                                       w);
                        }
                    }
                }
                if (std::isnan(nlooks))
                    break;
            }

            // ignoring boundary or low-sampled area elements
            if (std::isnan(nlooks) || 
                nlooks < isce::core::AREA_PROJECTION_MIN_VALID_SAMPLES_RATIO * 
                std::abs(w_total) ||
                    (!std::isnan(min_nlooks) && nlooks <= min_nlooks))
                continue;

            // save geo-edges
            if (out_geo_vertices != nullptr)
            {
                if (i == 0) {
                    out_geo_vertices_a(i, jj + 1) = (a01 - start) / pixazm;
                    out_geo_vertices_r(i, jj + 1) = (r01 - r0) / dr;
                }
                if (i == 0 && jj == 0) {
                    out_geo_vertices_a(i, jj) = (a00 - start) / pixazm;
                    out_geo_vertices_r(i, jj) = (r00 - r0) / dr;
                }
                if (jj == 0) {
                    out_geo_vertices_a(i + 1, jj) = (a10 - start) / pixazm;
                    out_geo_vertices_r(i + 1, jj) = (r10 - r0) / dr;
                }

                out_geo_vertices_a(i + 1, jj + 1) = (a11 - start) / pixazm;
                out_geo_vertices_r(i + 1, jj + 1) = (r11 - r0) / dr;
            }

            // save geo-edges
            if (out_dem_vertices != nullptr)
            {
                if (i == 0) {
                    out_dem_vertices_array(i, jj + 1) = dem01[2];
                }
                if (i == 0 && jj == 0) {
                    out_dem_vertices_array(i, jj) = dem00[2];
                }
                if (jj == 0) {
                    out_dem_vertices_array(i + 1, jj) = dem10[2];
                }

                out_dem_vertices_array(i + 1, jj + 1) = dem11[2];
            }

            // x, y positions are binned by integer quotient (floor)
            const int x = (int) jj / geogrid_upsampling;
            const int y = (int) i / geogrid_upsampling;

            if (output_mode ==
                    geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT) {
                area_total /= nlooks;
            } else {
                area_total = 1;
            }

            // save nlooks
            if (out_geo_nlooks != nullptr && std::isnan(out_geo_nlooks_array(y, x)))
                out_geo_nlooks_array(y, x) = (radar_grid_nlooks * nlooks);
            else if (out_geo_nlooks != nullptr)
                out_geo_nlooks_array(y, x) += (radar_grid_nlooks * nlooks);

            // save rtc
            if (out_geo_rtc != nullptr && std::isnan(out_geo_rtc_array(y, x)))
                out_geo_rtc_array(y, x) = (area_total/ (geogrid_upsampling *
                                   geogrid_upsampling));
            else if (out_geo_rtc != nullptr)
                out_geo_rtc_array(y, x) += (area_total/ (geogrid_upsampling *
                                   geogrid_upsampling)); 

            // divide by total and save result in the output array
            for (int band = 0; band < nbands; ++band)
                geoDataBlock[band].get()->operator()(y, x) =
                        (geoDataBlock[band].get()->operator()(y, x) +
                         ((T_out)((cumulative_sum[band]) *
                                  abs_cal_factor_effective /
                                  (nlooks * geogrid_upsampling *
                                   geogrid_upsampling))));
        }
    }

    for (int band = 0; band < nbands; ++band) {
        for (int i = 0; i < this_block_size; ++i)
            for (int jj = 0; jj < (int) _geoGridWidth; ++jj) {
                T_out geo_value = geoDataBlock[band].get()->operator()(i, jj);
                if (!std::isnan(clip_min) && std::abs(geo_value) < clip_min)
                    geoDataBlock[band].get()->operator()(i, jj) = clip_min;
                else if (!std::isnan(clip_max) && std::abs(geo_value) > clip_max)
                    geoDataBlock[band].get()->operator()(i, jj) = clip_max;
                else if (std::abs(geo_value) == 0)
                    geoDataBlock[band].get()->operator()(i, jj) =
                            std::numeric_limits<T_out>::quiet_NaN();
            }    
        #pragma omp critical
        {
            output_raster.setBlock(geoDataBlock[band].get()->data(), 0,
                                   block * block_size, _geoGridWidth,
                                   this_block_size, band + 1);
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

    if (out_dem_vertices != nullptr)
    #pragma omp critical
    {
        out_dem_vertices->setBlock(out_dem_vertices_array.data(), 0,
                                   block * block_size_with_upsampling, jmax + 1,
                                   this_block_size_with_upsampling + 1, 1);
    }


    if (out_geo_nlooks != nullptr)
    #pragma omp critical
    {
        out_geo_nlooks->setBlock(out_geo_nlooks_array.data(), 0,
                                 block * block_size, _geoGridWidth,
                                 this_block_size, 1);
    }

    if (out_geo_rtc != nullptr)
    #pragma omp critical
    {
        out_geo_rtc->setBlock(out_geo_rtc_array.data(), 0, block * block_size,
                              _geoGridWidth, this_block_size, 1);
    }
}

template class Geocode<float>;
template class Geocode<double>;
template class Geocode<std::complex<float>>;
template class Geocode<std::complex<double>>;


// template <typename T>
std::vector<float> getGeoAreaElementMean(
        const std::vector<double>& x_vect, const std::vector<double>& y_vect,
        const isce::product::RadarGridParameters& radar_grid,
        const isce::core::Orbit& orbit,
        const isce::core::LUT2d<double>& input_dop,
        isce::io::Raster& input_raster, isce::io::Raster& dem_raster,
        isce::geometry::rtcInputRadiometry input_radiometry, int exponent,
        geocodeOutputMode output_mode, double geogrid_upsampling,
        float rtc_min_value_db, double abs_cal_factor, float radar_grid_nlooks,
        float* out_nlooks, isce::core::dataInterpMethod interp_method,
        double threshold, int num_iter, double delta_range) {

    pyre::journal::info_t info("isce.geometry.getGeoAreaElementMean");

    if (std::isnan(geogrid_upsampling))
        geogrid_upsampling = 1;
    assert(geogrid_upsampling > 0);

    const double x0 = std::min_element(x_vect.begin(), x_vect.end())[0];
    const double xf = std::max_element(x_vect.begin(), x_vect.end())[0];

    const double y0 = std::min_element(y_vect.begin(), y_vect.end())[0];
    const double yf = std::max_element(y_vect.begin(), y_vect.end())[0];

    const double dx = dem_raster.dx();
    const double dy = dem_raster.dy();

    std::string input_radiometry_str =
            get_input_radiometry_str(input_radiometry);

    info << "input radiometry: " << input_radiometry_str
         << pyre::journal::newline << "look side: " << radar_grid.lookSide()
         << pyre::journal::newline
         << "radar_grid length: " << radar_grid.length()
         << ", width: " << radar_grid.width() << pyre::journal::newline
         << "RTC min value [dB]: " << rtc_min_value_db << pyre::journal::endl;

    const double margin_x = std::abs(dx) * 10;
    const double margin_y = std::abs(dy) * 10;
    DEMInterpolator dem_interp;

    dem_interp.loadDEM(dem_raster, x0 - margin_x, xf + margin_x,
                       std::min(y0, yf) - margin_y,
                       std::max(y0, yf) + margin_y);

    isce::core::Ellipsoid ellipsoid =
            isce::core::Ellipsoid(isce::core::EarthSemiMajorAxis,
                                  isce::core::EarthEccentricitySquared);

    int epsg = dem_raster.getEPSG();
    std::unique_ptr<isce::core::ProjectionBase> proj(
            isce::core::createProj(epsg));
    const double pixazm = radar_grid.azimuthTimeInterval();
    const double start = radar_grid.sensingStart() - 0.5 * pixazm;
    const double dr = radar_grid.rangePixelSpacing();
    double r0 = radar_grid.startingRange() - 0.5 * dr;

    double a = radar_grid.sensingMid();
    double r = radar_grid.midRange();

    if (x_vect.size() != y_vect.size()) {
        std::string error_message =
                "ERROR x and y vectors have a different number of elements.";
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), error_message);
    }
    int n_elements = x_vect.size();
    std::vector<double> a_vect, r_vect;
    a_vect.reserve(n_elements);
    r_vect.reserve(n_elements);

    info << "polygon indexes (a, r): ";

    for (int i = 0; i < n_elements; ++i) {

        const double x = x_vect[i];
        const double y = y_vect[i];

        const Vec3 dem11 = {x, y, dem_interp.interpolateXY(x, y)};
        int converged =
                geo2rdr(proj->inverse(dem11), ellipsoid, orbit, input_dop, a, r,
                        radar_grid.wavelength(), radar_grid.lookSide(),
                        threshold, num_iter, delta_range);
        if (!converged) {
            info << "WARNING convergence not found for vertex (x, y): " << x
                 << ", " << y << pyre::journal::endl;
            continue;
        }
        double a_index = (a - start) / pixazm;
        double r_index = (r - r0) / dr;

        info << "(" << a_index << ", " << r_index << "), ";

        a_vect.emplace_back(a_index);
        r_vect.emplace_back(r_index);
    }

    info << pyre::journal::endl;

    const double a_min = std::min_element(a_vect.begin(), a_vect.end())[0];
    const double a_max = std::max_element(a_vect.begin(), a_vect.end())[0];

    const double r_min = std::min_element(r_vect.begin(), r_vect.end())[0];
    const double r_max = std::max_element(r_vect.begin(), r_vect.end())[0];

    const int y_min = std::max(0, (int) std::floor(a_min));
    const int x_min = std::max(0, (int) std::floor(r_min));
    const int ysize =
            std::min((int) radar_grid.length(), (int) std::ceil(a_max)) - y_min;
    const int xsize =
            std::min((int) radar_grid.width(), (int) std::ceil(r_max)) - x_min;

    info << "cropping radar grid from index (a0: " << y_min;
    info << ", r0: " << x_min << ") to index (af: " << y_min + ysize;
    info << ", rf: " << x_min + xsize << ")" << pyre::journal::endl; 

    isce::product::RadarGridParameters radar_grid_cropped =
            radar_grid.offsetAndResize(y_min, x_min, ysize, xsize);

    info << "cropped radar_grid length: " << radar_grid_cropped.length()
            << ", width: " << radar_grid_cropped.width() << pyre::journal::newline;


    if (output_mode == geocodeOutputMode::INTERP) {
        std::string error_msg = "invalid option";
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    if (abs_cal_factor != 1)
        info << "absolute calibration factor: " << abs_cal_factor
            << pyre::journal::endl;

    if (output_mode == geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT) {

        std::string input_radiometry_str =
                get_input_radiometry_str(input_radiometry);
        info << "input radiometry: " << input_radiometry_str
             << pyre::journal::endl;
    }

    isce::core::Matrix<float> rtc_area;
    std::unique_ptr<isce::io::Raster> rtc_raster_unique_ptr;
    if (output_mode == geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT) {

        info << "computing RTC area factor..." << pyre::journal::endl;
        rtc_raster_unique_ptr = std::make_unique<isce::io::Raster>(
                "/vsimem/dummy", radar_grid_cropped.width(),
                radar_grid_cropped.length(), 1, GDT_Float32, "ENVI");
        isce::geometry::rtcAreaMode rtc_area_mode =
                isce::geometry::rtcAreaMode::AREA_FACTOR;
        isce::geometry::rtcAlgorithm rtc_algorithm =
                isce::geometry::rtcAlgorithm::RTC_AREA_PROJECTION;

        isce::geometry::rtcMemoryMode rtc_memory_mode =
                isce::geometry::rtcMemoryMode::RTC_SINGLE_BLOCK;

        facetRTC(radar_grid_cropped, orbit, input_dop, dem_raster,
                 *rtc_raster_unique_ptr.get(), input_radiometry, rtc_area_mode,
                 rtc_algorithm, geogrid_upsampling * 2, rtc_min_value_db,
                 radar_grid_nlooks, nullptr, rtc_memory_mode,
                 interp_method, threshold, num_iter, delta_range);

        rtc_area.resize(radar_grid_cropped.length(),
                        radar_grid_cropped.width());

        rtc_raster_unique_ptr.get()->getBlock(rtc_area.data(), 0, 0,
                                              radar_grid_cropped.width(),
                                              radar_grid_cropped.length(), 1);

        info << "... done (RTC) " << pyre::journal::endl;
    }

    double rtc_min_value = 0;
    if (!std::isnan(rtc_min_value_db) &&
        output_mode == geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT) {
        rtc_min_value = std::pow(10, (rtc_min_value_db / 10));
        info << "RTC min. value: " << rtc_min_value_db
             << " [dB] = " << rtc_min_value << pyre::journal::endl;
    }

    GDALDataType input_dtype = input_raster.dtype();
    if (exponent == 0 && GDALDataTypeIsComplex(input_dtype))
        exponent = 2;

    if (input_raster.dtype() == GDT_Float32) {
        info << "dtype: GDT_Float32" << pyre::journal::endl;
        return _getGeoAreaElementMean<float>(
                r_vect, a_vect, x_min, y_min, rtc_area, radar_grid_cropped,
                input_raster, output_mode, rtc_min_value, out_nlooks,
                abs_cal_factor, radar_grid_nlooks);
    } else if (input_raster.dtype() == GDT_CFloat32) {
        info << "dtype: GDT_CFloat32" << pyre::journal::endl;
        return _getGeoAreaElementMean<std::complex<float>>(
                r_vect, a_vect, x_min, y_min, rtc_area, radar_grid_cropped,
                input_raster, output_mode, rtc_min_value, out_nlooks,
                abs_cal_factor, radar_grid_nlooks);
    } else
        info << "ERROR not implemented for datatype: " << input_raster.dtype()
             << pyre::journal::endl;

    std::vector<float> empty_vector;
    return empty_vector;
}

template<typename T>
std::vector<float> _getGeoAreaElementMean(
        const std::vector<double>& r_vect, const std::vector<double>& a_vect,
        int x_min, int y_min, isce::core::Matrix<float>& rtc_area,
        const isce::product::RadarGridParameters& radar_grid,
        isce::io::Raster& input_raster, geocodeOutputMode output_mode,
        float rtc_min_value, float* out_nlooks,
        double abs_cal_factor, float radar_grid_nlooks) {

    pyre::journal::info_t info("isce.geometry._getGeoAreaElementMean");

    // number of bands in the input raster
    const int nbands = input_raster.numBands();
    const int size_y = radar_grid.length();
    const int size_x = radar_grid.width();
    info << "nbands: " << nbands << pyre::journal::endl;

    std::vector<std::unique_ptr<isce::core::Matrix<T>>> rdrDataBlock;
    rdrDataBlock.reserve(nbands);

    for (int band = 0; band < nbands; ++band) {
        if (nbands == 1)
            info << "loading slant-range image..." << pyre::journal::endl;
        else
            info << "loading slant-range band: " << band << pyre::journal::endl;
        rdrDataBlock.emplace_back(
                std::make_unique<isce::core::Matrix<T>>(size_y, size_x));

        input_raster.getBlock(rdrDataBlock[band].get()->data(), x_min, y_min,
                              size_x, size_y, band + 1);
    }

    isce::core::Matrix<double> w_arr(size_y, size_x);
    w_arr.fill(0);

    int plane_orientation;
    if (radar_grid.lookSide() == isce::core::LookSide::Left)
        plane_orientation = -1;
    else
        plane_orientation = 1;

    double w_total = 0;
    for (int i = 0; i < r_vect.size(); ++i) {

        double x00, y00, x01, y01;
        y00 = a_vect[i];
        x00 = r_vect[i];

        if (i < r_vect.size() - 1) {
            y01 = a_vect[i + 1];
            x01 = r_vect[i + 1];
        } else {
            y01 = a_vect[0];
            x01 = r_vect[0];
        }
        areaProjIntegrateSegment(y00 - y_min, y01 - y_min, x00 - x_min,
                                 x01 - x_min, size_y, size_x, w_arr, w_total,
                                 plane_orientation);
    }
    std::vector<float> cumulative_sum(nbands);
    double nlooks = 0;

    Geocode<T> geo_obj;
    for (int y = 0; y < size_y; ++y)
        for (int x = 0; x < size_x; ++x) {
            double w = w_arr(y, x);
            if (w == 0)
                continue;
            if (output_mode ==
                geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT) {
                const float rtc_value = rtc_area(y, x);
                if (std::isnan(rtc_value) || rtc_value < rtc_min_value)
                    continue;
                nlooks += w;
                w /= rtc_value;
            } else {
                nlooks += w;
            }

            for (int band = 0; band < nbands; ++band)
                _accumulate(cumulative_sum[band],
                            rdrDataBlock[band].get()->operator()(y, x), w);
        }

    info << "nlooks: " << radar_grid_nlooks * std::abs(nlooks)
         << pyre::journal::endl;
    for (int band = 0; band < nbands; ++band) {
        cumulative_sum[band] =
                (cumulative_sum[band] * abs_cal_factor / nlooks);
        info << "mean value (band = " << band << "): " << cumulative_sum[band]
             << pyre::journal::endl;
    }


    if (out_nlooks != nullptr) {
        *out_nlooks = radar_grid_nlooks * std::abs(nlooks);
    }

    return cumulative_sum;
}





} // namespace geometry
} // namespace isce
