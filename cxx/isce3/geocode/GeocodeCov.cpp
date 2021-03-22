#include "GeocodeCov.h"

#include <algorithm>
#include <cmath>
#include <cpl_virtualmem.h>
#include <limits>
#include <chrono> 

#include <isce3/core/Basis.h>
#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Projections.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/RTC.h>
#include <isce3/geometry/boundingbox.h>
#include <isce3/geometry/geometry.h>
#include <isce3/signal/Looks.h>
#include <isce3/signal/signalUtils.h>
#include <isce3/core/TypeTraits.h>

using isce3::core::OrbitInterpBorderMode;
using isce3::core::Vec3;

namespace isce3 { namespace geocode {

template<typename T1, typename T2>
auto operator*(const std::complex<T1>& lhs, const T2& rhs)
{
    using U = typename std::common_type_t<T1, T2>;
    return std::complex<U>(lhs) * U(rhs);
}

template<typename T1, typename T2>
auto operator*(const T1& lhs, const std::complex<T2>& rhs)
{
    using U = typename std::common_type_t<T1, T2>;
    return U(lhs) * std::complex<U>(rhs);
}

template<typename T, typename T_out>
void _convertToOutputType(T a, T_out& b)
{
    b = a;
}

template<typename T, typename T_out>
void _convertToOutputType(std::complex<T> a, T_out& b)
{
    b = std::norm(a);
}

template<typename T, typename T_out>
void _convertToOutputType(std::complex<T> a, std::complex<T_out>& b)
{
    b = a;
}

template<typename T, typename T_out>
void _accumulate(T_out& band_value, T a, double b)
{
    if (b == 0)
        return;
    T_out a2;
    _convertToOutputType(a, a2);
    band_value += a2 * b;
}

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

template<class T>
void Geocode<T>::geocode(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
        isce3::io::Raster& dem_raster, geocodeOutputMode output_mode,
        double geogrid_upsampling, bool flag_upsample_radar_grid,
        bool flag_apply_rtc,
        isce3::geometry::rtcInputTerrainRadiometry input_terrain_radiometry,
        isce3::geometry::rtcOutputTerrainRadiometry output_terrain_radiometry,
        int exponent, float rtc_min_value_db, double rtc_geogrid_upsampling,
        isce3::geometry::rtcAlgorithm rtc_algorithm, double abs_cal_factor,
        float clip_min, float clip_max, float min_nlooks,
        float radar_grid_nlooks, isce3::io::Raster* out_off_diag_terms,
        isce3::io::Raster* out_geo_rdr,
        isce3::io::Raster* out_geo_dem, isce3::io::Raster* out_geo_nlooks,
        isce3::io::Raster* out_geo_rtc, isce3::io::Raster* input_rtc,
        isce3::io::Raster* output_rtc, geocodeMemoryMode geocode_memory_mode,
        const int min_block_size, const int max_block_size,
        isce3::core::dataInterpMethod dem_interp_method)
{
    bool flag_complex_to_real = isce3::signal::verifyComplexToRealCasting(
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
        geocodeAreaProj<T>(
                radar_grid, input_raster, output_raster, dem_raster,
                output_mode, geogrid_upsampling, flag_upsample_radar_grid,
                flag_apply_rtc,
                input_terrain_radiometry, output_terrain_radiometry,
                rtc_min_value_db, rtc_geogrid_upsampling,
                rtc_algorithm, abs_cal_factor, clip_min, clip_max, min_nlooks,
                radar_grid_nlooks, out_off_diag_terms, out_geo_rdr,
                out_geo_dem, out_geo_nlooks, out_geo_rtc, input_rtc,
                output_rtc, geocode_memory_mode, min_block_size,
                max_block_size, dem_interp_method);
    else if (std::is_same<T, double>::value ||
             std::is_same<T, std::complex<double>>::value)
        geocodeAreaProj<double>(
                radar_grid, input_raster, output_raster, dem_raster,
                output_mode, geogrid_upsampling, flag_upsample_radar_grid,
                flag_apply_rtc,
                input_terrain_radiometry, output_terrain_radiometry, 
                rtc_min_value_db, rtc_geogrid_upsampling,
                rtc_algorithm, abs_cal_factor, clip_min, clip_max, min_nlooks,
                radar_grid_nlooks, out_off_diag_terms, out_geo_rdr,
                out_geo_dem, out_geo_nlooks, out_geo_rtc, input_rtc,
                output_rtc, geocode_memory_mode, min_block_size,
                max_block_size, dem_interp_method);
    else
        geocodeAreaProj<float>(
                radar_grid, input_raster, output_raster, dem_raster,
                output_mode, geogrid_upsampling, flag_upsample_radar_grid,
                flag_apply_rtc,
                input_terrain_radiometry, output_terrain_radiometry, 
                rtc_min_value_db, rtc_geogrid_upsampling,
                rtc_algorithm, abs_cal_factor, clip_min, clip_max, min_nlooks,
                radar_grid_nlooks, out_off_diag_terms, out_geo_rdr,
                out_geo_dem, out_geo_nlooks, out_geo_rtc, input_rtc,
                output_rtc, geocode_memory_mode, min_block_size,
                max_block_size, dem_interp_method);
}

template<class T>
template<class T_out>
void Geocode<T>::geocodeInterp(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& inputRaster, isce3::io::Raster& outputRaster,
        isce3::io::Raster& demRaster)
{

    pyre::journal::info_t info("isce.geocode.GeocodeCov.geocodeInterp");
    auto start_time = std::chrono::high_resolution_clock::now();

    std::unique_ptr<isce3::core::Interpolator<T_out>> interp {
            isce3::core::createInterpolator<T_out>(_data_interp_method)};

    // number of bands in the input raster
    int nbands = inputRaster.numBands();

    // create projection based on _epsg code
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(_epsgOut));

    // instantiate the DEMInterpolator
    isce3::geometry::DEMInterpolator demInterp;

    // Compute number of blocks in the output geocoded grid
    int nBlocks = _geoGridLength / _linesPerBlock;
    if ((_geoGridLength % _linesPerBlock) != 0)
        nBlocks += 1;

    info << "nBlocks: " << nBlocks << pyre::journal::newline;
    // loop over the blocks of the geocoded Grid
    for (int block = 0; block < nBlocks; ++block) {
        info << "block: " << block << pyre::journal::endl;
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
                azimuthFirstLine =
                        std::min(azimuthFirstLine, localAzimuthFirstLine);
                azimuthLastLine =
                        std::max(azimuthLastLine, localAzimuthLastLine);
                rangeFirstPixel =
                        std::min(rangeFirstPixel, localRangeFirstPixel);
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
        isce3::core::Matrix<T_out> rdrDataBlock(rdrBlockLength, rdrBlockWidth);
        isce3::core::Matrix<T_out> geoDataBlock(geoBlockLength, _geoGridWidth);

        // set NaN values according to T_out, i.e. real (NaN) or complex (NaN,
        // NaN)
        using T_out_real = typename isce3::real<T_out>::type;
        T_out nan_t_out = 0;
        nan_t_out *= std::numeric_limits<T_out_real>::quiet_NaN();

        // fill both matrices with NaN
        rdrDataBlock.fill(nan_t_out);
        geoDataBlock.fill(nan_t_out);

        // for each band in the input:
        for (int band = 0; band < nbands; ++band) {
            info << "band: " << band << pyre::journal::newline;
            // get a block of data
            info << "get data block " << pyre::journal::endl;
            if ((std::is_same<T, std::complex<float>>::value ||
                 std::is_same<T, std::complex<double>>::value) &&
                (std::is_same<T_out, float>::value ||
                 std::is_same<T_out, double>::value)) {
                isce3::core::Matrix<T> rdrDataBlockTemp(rdrBlockLength,
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
            info << "interpolate " << pyre::journal::newline;
            _interpolate(rdrDataBlock, geoDataBlock, radarX, radarY,
                         rdrBlockWidth, rdrBlockLength, azimuthFirstLine,
                         rangeFirstPixel, interp.get());

            // set output
            info << "set output " << pyre::journal::endl;
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

    auto elapsed_time_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    float elapsed_time = ((float) elapsed_time_milliseconds.count()) / 1e3;
    info << "elapsed time (GEO-IN) [s]: " << elapsed_time << pyre::journal::endl;    
}

template<class T>
template<class T_out>
void Geocode<T>::_interpolate(isce3::core::Matrix<T_out>& rdrDataBlock,
                              isce3::core::Matrix<T_out>& geoDataBlock,
                              std::valarray<double>& radarX,
                              std::valarray<double>& radarY,
                              int radarBlockWidth, int radarBlockLength,
                              int azimuthFirstLine, int rangeFirstPixel,
                              isce3::core::Interpolator<T_out>* _interp)
{
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
void Geocode<T>::_loadDEM(isce3::io::Raster& demRaster,
                          isce3::geometry::DEMInterpolator& demInterp,
                          isce3::core::ProjectionBase* proj, int lineStart,
                          int blockLength, int blockWidth, double demMargin)
{
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
        std::unique_ptr<isce3::core::ProjectionBase> demproj(
                isce3::core::createProj(epsgcode));

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
    demMargin = (epsgcode != 4326) ? isce3::core::decimaldeg2meters(demMargin)
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
void Geocode<T>::_geo2rdr(const isce3::product::RadarGridParameters& radar_grid,
                          double x, double y, double& azimuthTime,
                          double& slantRange,
                          isce3::geometry::DEMInterpolator& demInterp,
                          isce3::core::ProjectionBase* proj)
{
    // coordinate in the output projection system
    const Vec3 xyz {x, y, 0.0};

    // transform the xyz in the output projection system to llh
    Vec3 llh = proj->inverse(xyz);

    // interpolate the height from the DEM for this pixel
    llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

    // Perform geo->rdr iterations
    int geostat = isce3::geometry::geo2rdr(
            llh, _ellipsoid, _orbit, _doppler, azimuthTime, slantRange,
            radar_grid.wavelength(), radar_grid.lookSide(), _threshold,
            _numiter, 1.0e-8);

    // Check convergence
    if (geostat == 0) {
        azimuthTime = std::numeric_limits<double>::quiet_NaN();
        slantRange = std::numeric_limits<double>::quiet_NaN();
        return;
    }
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
        geocodeMemoryMode geocode_memory_mode, pyre::journal::info_t& info)
{
    int nbands = input_raster.numBands();
    rdrData.reserve(nbands);
    for (int band = 0; band < nbands; ++band) {
        if (geocode_memory_mode !=
            geocodeMemoryMode::BLOCKS_GEOGRID_AND_RADARGRID) {
            info << "reading input raster band: " << band + 1
                 << pyre::journal::endl;
        }
        rdrData.emplace_back(
                std::make_unique<isce3::core::Matrix<T_out>>(size_y, size_x));
        if (!flag_upsample_radar_grid && std::is_same<T, T_out>::value) {
#pragma omp critical
            input_raster.getBlock(rdrData[band]->data(), xidx, yidx, size_x,
                                  size_y, band + 1);
        } else if (!flag_upsample_radar_grid) {
            isce3::core::Matrix<T> radar_data_in(size_y, size_x);
#pragma omp critical
            input_raster.getBlock(radar_data_in.data(), xidx, yidx, size_x,
                                  size_y, band + 1);

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
            std::string error_msg = "radar-grid upsampling is only available";
            error_msg += " for complex inputs";
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
        } else {
            int radargrid_nblocks, radar_block_size;
            if (geocode_memory_mode == geocodeMemoryMode::SINGLE_BLOCK ||
                geocode_memory_mode ==
                        geocodeMemoryMode::BLOCKS_GEOGRID_AND_RADARGRID) {
                radargrid_nblocks = 1;
                radar_block_size = size_y;

            } else {
                isce3::geometry::areaProjGetNBlocks(
                        size_y, size_x, nbands, sizeof(T), nullptr, 1,
                        &radar_block_size, nullptr, &radargrid_nblocks);
            }
            if (radargrid_nblocks == 1) {
                _processUpsampledBlock<T, T_out>(
                        rdrData[band].get(), 0, radar_block_size, input_raster,
                        xidx, yidx, size_x, size_y, band);
            } else {
                // #pragma omp parallel for schedule(dynamic)
                for (size_t block = 0; block < (size_t) radargrid_nblocks;
                     ++block) {
                    _processUpsampledBlock<T, T_out>(
                            rdrData[band].get(), block, radar_block_size,
                            input_raster, xidx, yidx, size_x, size_y, band);
                }
            }
        }
    }
}

template<class T>
template<class T_out>
void Geocode<T>::geocodeAreaProj(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
        isce3::io::Raster& dem_raster, geocodeOutputMode output_mode,
        double geogrid_upsampling, bool flag_upsample_radar_grid,
        bool flag_apply_rtc,
        isce3::geometry::rtcInputTerrainRadiometry input_terrain_radiometry,
        isce3::geometry::rtcOutputTerrainRadiometry output_terrain_radiometry,
        float rtc_min_value_db, double rtc_geogrid_upsampling,
        isce3::geometry::rtcAlgorithm rtc_algorithm, double abs_cal_factor,
        float clip_min, float clip_max, float min_nlooks,
        float radar_grid_nlooks, isce3::io::Raster* out_off_diag_terms,
        isce3::io::Raster* out_geo_rdr,
        isce3::io::Raster* out_geo_dem, isce3::io::Raster* out_geo_nlooks,
        isce3::io::Raster* out_geo_rtc, isce3::io::Raster* input_rtc,
        isce3::io::Raster* output_rtc, geocodeMemoryMode geocode_memory_mode,
        const int min_block_size, const int max_block_size,
        isce3::core::dataInterpMethod dem_interp_method)
{

    if (std::isnan(geogrid_upsampling))
        geogrid_upsampling = 1;
    assert(geogrid_upsampling > 0);
    assert(output_mode != geocodeOutputMode::INTERP);

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
        geocodeAreaProj<T_out>(
                upsampled_radar_grid, input_raster, output_raster, dem_raster,
                output_mode, geogrid_upsampling, flag_upsample_radar_grid,
                flag_apply_rtc,
                input_terrain_radiometry, output_terrain_radiometry,
                rtc_min_value_db, rtc_geogrid_upsampling, rtc_algorithm, 
                abs_cal_factor, clip_min, clip_max, min_nlooks, 
                upsampled_radar_grid_nlooks, out_off_diag_terms, out_geo_rdr, 
                out_geo_dem, out_geo_nlooks, out_geo_rtc, input_rtc, output_rtc,
                geocode_memory_mode, min_block_size, max_block_size,
                dem_interp_method);
        return;
    }
    pyre::journal::info_t info("isce.geometry.Geocode.geocodeAreaProj");
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
        assert(isce3::is_complex<T>());
        assert(GDALDataTypeIsComplex(out_off_diag_terms->dtype()));
    } else {
        info << "nbands: " << nbands << pyre::journal::newline;
        info << "full covariance: false" << pyre::journal::newline;
    }

    if (flag_apply_rtc) {
        std::string input_terrain_radiometry_str =
                get_input_terrain_radiometry_str(input_terrain_radiometry);
        info << "input terrain radiometry: " << input_terrain_radiometry_str
             << pyre::journal::newline;
        std::string output_terrain_radiometry_str =
                get_output_terrain_radiometry_str(output_terrain_radiometry);
        info << "output terrain radiometry: " << output_terrain_radiometry_str
             << pyre::journal::newline;
    }
    if (!std::isnan(clip_min))
        info << "clip min: " << clip_min << pyre::journal::newline;

    if (!std::isnan(clip_max))
        info << "clip max: " << clip_max << pyre::journal::newline;

    if (!std::isnan(min_nlooks))
        info << "nlooks min: " << min_nlooks << pyre::journal::newline;

    isce3::core::Matrix<float> rtc_area;
    if (flag_apply_rtc) {

        // declare pointer to the raster containing the RTC area factor
        isce3::io::Raster* rtc_raster;
        std::unique_ptr<isce3::io::Raster> rtc_raster_unique_ptr;

        if (input_rtc == nullptr) {

            info << "calling RTC (from geocode)..." << pyre::journal::endl;

            // if RTC (area factor) raster does not needed to be saved,
            // initialize it as a GDAL memory virtual file
            if (output_rtc == nullptr) {
                rtc_raster_unique_ptr = std::make_unique<isce3::io::Raster>(
                        "/vsimem/dummy", radar_grid.width(),
                        radar_grid.length(), 1, GDT_Float32, "ENVI");
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

            isce3::geometry::rtcMemoryMode rtc_memory_mode;
            if (geocode_memory_mode == geocodeMemoryMode::AUTO)
                rtc_memory_mode = isce3::geometry::RTC_AUTO;
            else if (geocode_memory_mode == geocodeMemoryMode::SINGLE_BLOCK)
                rtc_memory_mode = isce3::geometry::RTC_SINGLE_BLOCK;
            else
                rtc_memory_mode = isce3::geometry::RTC_BLOCKS_GEOGRID;

            computeRtc(dem_raster, *rtc_raster, radar_grid, _orbit, _doppler,
                    _geoGridStartY, _geoGridSpacingY, _geoGridStartX,
                    _geoGridSpacingX, _geoGridLength, _geoGridWidth, _epsgOut,
                    input_terrain_radiometry, output_terrain_radiometry,
                    rtc_area_mode, rtc_algorithm, rtc_geogrid_upsampling,
                    rtc_min_value_db, radar_grid_nlooks, nullptr, nullptr,
                    nullptr, rtc_memory_mode, dem_interp_method, _threshold,
                    _numiter, 1.0e-8);
        } else {
            info << "reading pre-computed RTC..." << pyre::journal::newline;
            rtc_raster = input_rtc;
        }

        rtc_area.resize(radar_grid.length(), radar_grid.width());
        rtc_raster->getBlock(rtc_area.data(), 0, 0, radar_grid.width(),
                             radar_grid.length(), 1);
    }

    // number of bands in the input raster

    info << "nbands: " << nbands << pyre::journal::newline;

    // create projection based on epsg code
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(_epsgOut));

    // start (az) and r0 at the outer edge of the first pixel:
    const double pixazm = radar_grid.azimuthTimeInterval();
    const double start = radar_grid.sensingStart() - 0.5 * pixazm;
    const double dr = radar_grid.rangePixelSpacing();
    double r0 = radar_grid.startingRange() - 0.5 * dr;

    // Bounds for valid RDC coordinates

    const int imax = _geoGridLength * geogrid_upsampling;
    const int jmax = _geoGridWidth * geogrid_upsampling;

    info << "radar grid width: " << radar_grid.width()
         << ", length: " << radar_grid.length() << pyre::journal::newline;

    info << "geogrid upsampling: " << geogrid_upsampling << pyre::journal::newline;

    int epsgcode = dem_raster.getEPSG();

    if (epsgcode < 0) {
        std::string error_msg = "invalid DEM EPSG";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

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

    bool is_radar_grid_single_block =
            (geocode_memory_mode !=
             geocodeMemoryMode::BLOCKS_GEOGRID_AND_RADARGRID);

    std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>> rdrData;
    std::vector<std::unique_ptr<isce3::core::Matrix<T>>> rdrDataT;

    /*
    T - input data template;
    T2 - input template for the _runBlock, that is equal to
         the off-diagonal terms template (if
         applicable) or T2 == T_out (otherwise);
    T_out - diagonal terms template.
    */

    if (is_radar_grid_single_block) {
        if (std::is_same<T, T_out>::value || nbands_off_diag_terms > 0) {
            _getUpsampledBlock<T, T>(rdrDataT, input_raster, 0, 0,
                                     radar_grid.width(), radar_grid.length(),
                                     flag_upsample_radar_grid,
                                     geocode_memory_mode, info);
        } else {
            _getUpsampledBlock<T, T_out>(
                    rdrData, input_raster, 0, 0, radar_grid.width(),
                    radar_grid.length(), flag_upsample_radar_grid,
                    geocode_memory_mode, info);
        }
    }
    int block_size_x, nblocks_x, block_size_with_upsampling_x;
    int block_size_y, nblocks_y, block_size_with_upsampling_y;
    if (geocode_memory_mode == geocodeMemoryMode::SINGLE_BLOCK) {

        nblocks_x = 1;
        block_size_x = _geoGridWidth;
        block_size_with_upsampling_x = jmax;

        nblocks_y = 1;
        block_size_y = _geoGridLength;
        block_size_with_upsampling_y = imax;
    } else {
        isce3::geometry::areaProjGetNBlocks(
                imax, jmax, nbands + nbands_off_diag_terms, sizeof(T_out),
                &info, geogrid_upsampling, &block_size_with_upsampling_y,
                &block_size_y, &nblocks_y, &block_size_with_upsampling_x,
                &block_size_x, &nblocks_x, min_block_size, max_block_size);
    }

    long long numdone = 0;

    info << "starting geocoding" << pyre::journal::endl;
    if (!std::is_same<T, T_out>::value && nbands_off_diag_terms == 0) {
#pragma omp parallel for schedule(dynamic)
        for (int block_y = 0; block_y < nblocks_y; ++block_y) {
            for (int block_x = 0; block_x < nblocks_x; ++block_x) {
                _runBlock<T_out, T_out>(
                        radar_grid, is_radar_grid_single_block, rdrData,
                        block_size_y, block_size_with_upsampling_y, block_y,
                        block_size_x, block_size_with_upsampling_x, block_x,
                        numdone, progress_block, geogrid_upsampling, nbands,
                        nbands_off_diag_terms, dem_interp_method, dem_raster,
                        out_off_diag_terms, out_geo_rdr, out_geo_dem,
                        out_geo_nlooks, out_geo_rtc, start, pixazm, dr, r0,
                        proj.get(), flag_apply_rtc,
                        rtc_area, input_raster, output_raster,
                        rtc_min_value, abs_cal_factor, clip_min,
                        clip_max, min_nlooks, radar_grid_nlooks,
                        flag_upsample_radar_grid, geocode_memory_mode, info);
            }
        }
    } else {
#pragma omp parallel for schedule(dynamic)
        for (int block_y = 0; block_y < nblocks_y; ++block_y) {
            for (int block_x = 0; block_x < nblocks_x; ++block_x) {
                _runBlock<T, T_out>(
                        radar_grid, is_radar_grid_single_block, rdrDataT,
                        block_size_y, block_size_with_upsampling_y, block_y,
                        block_size_x, block_size_with_upsampling_x, block_x,
                        numdone, progress_block, geogrid_upsampling, nbands,
                        nbands_off_diag_terms, dem_interp_method, dem_raster,
                        out_off_diag_terms, out_geo_rdr, out_geo_dem,
                        out_geo_nlooks, out_geo_rtc, start, pixazm, dr, r0,
                        proj.get(), flag_apply_rtc,
                        rtc_area, input_raster, output_raster,
                        rtc_min_value, abs_cal_factor, clip_min,
                        clip_max, min_nlooks, radar_grid_nlooks,
                        flag_upsample_radar_grid, geocode_memory_mode, info);
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

    if (out_off_diag_terms != nullptr) {
        out_off_diag_terms->setGeoTransform(geotransform);
        out_off_diag_terms->setEPSG(_epsgOut);
    }

    auto elapsed_time_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    float elapsed_time = ((float) elapsed_time_milliseconds.count()) / 1e3;
    info << "elapsed time (GEO-AP) [s]: " << elapsed_time << pyre::journal::endl;     
}

template<class T>
void Geocode<T>::_getRadarPositionVect(
        double dem_pos_1, const int k_start, const int k_end,
        double geogrid_upsampling, double& a11, double& r11, double& a_min,
        double& r_min, double& a_max, double& r_max,
        std::vector<double>& a_vect, std::vector<double>& r_vect,
        std::vector<Vec3>& dem_vect,
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::core::ProjectionBase* proj,
        isce3::geometry::DEMInterpolator& dem_interp_block,
        bool flag_direction_line)
{

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
        int converged = isce3::geometry::geo2rdr(
                proj->inverse(dem11), _ellipsoid, _orbit, _doppler, a11, r11,
                radar_grid.wavelength(), radar_grid.lookSide(), _threshold,
                _numiter, 1.0e-8);

        // if it didn't converge, reset initial solution and continue
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

template<class T>
template<class T2, class T_out>
void Geocode<T>::_runBlock(
        const isce3::product::RadarGridParameters& radar_grid,
        bool is_radar_grid_single_block,
        std::vector<std::unique_ptr<isce3::core::Matrix<T2>>>& rdrData,
        int block_size_y, int block_size_with_upsampling_y, int block_y,
        int block_size_x, int block_size_with_upsampling_x, int block_x,
        long long& numdone, const long long& progress_block, 
        double geogrid_upsampling, int nbands,
        int nbands_off_diag_terms, isce3::core::dataInterpMethod dem_interp_method,
        isce3::io::Raster& dem_raster, isce3::io::Raster* out_off_diag_terms,
        isce3::io::Raster* out_geo_rdr,
        isce3::io::Raster* out_geo_dem, isce3::io::Raster* out_geo_nlooks,
        isce3::io::Raster* out_geo_rtc, const double start, const double pixazm,
        const double dr, double r0, isce3::core::ProjectionBase* proj,
        bool flag_apply_rtc,
        isce3::core::Matrix<float>& rtc_area, isce3::io::Raster& input_raster,
        isce3::io::Raster& output_raster,
        float rtc_min_value, double abs_cal_factor, float clip_min,
        float clip_max, float min_nlooks, float radar_grid_nlooks,
        bool flag_upsample_radar_grid, geocodeMemoryMode geocode_memory_mode,
        pyre::journal::info_t& info)
{

    // set NaN values according to T_out, i.e. real (NaN) or complex (NaN, NaN)
    using T_out_real = typename isce3::real<T_out>::type;
    T_out nan_t_out = 0;
    nan_t_out *= std::numeric_limits<T_out_real>::quiet_NaN();

    double abs_cal_factor_effective;
    if (!isce3::is_complex<T_out>())
        abs_cal_factor_effective = abs_cal_factor;
    else
        abs_cal_factor_effective = std::sqrt(abs_cal_factor);

    int radar_grid_range_upsampling = 1;
    if (flag_upsample_radar_grid)
        radar_grid_range_upsampling = 2;

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

    int ii_0 = block_y * block_size_with_upsampling_y;
    int jj_0 = block_x * block_size_with_upsampling_x;

    isce3::geometry::DEMInterpolator dem_interp_block(0, dem_interp_method);

    double minX = _geoGridStartX +
                  (((double) jj_0) / geogrid_upsampling * _geoGridSpacingX);
    double maxX =
            _geoGridStartX +
            std::min(((double) jj_0) / geogrid_upsampling + this_block_size_x,
                     (double) _geoGridWidth) *
                    _geoGridSpacingX;

    double minY = _geoGridStartY +
                  (((double) ii_0) / geogrid_upsampling * _geoGridSpacingY);
    double maxY = _geoGridStartY +
                  std::min(((double) ii_0) / geogrid_upsampling + this_block_size_y,
                           (double) _geoGridLength) *
                          _geoGridSpacingY;

    std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>> geoDataBlock;
    geoDataBlock.reserve(nbands);
    for (int band = 0; band < nbands; ++band)
        geoDataBlock.emplace_back(std::make_unique<isce3::core::Matrix<T_out>>(
                this_block_size_y, this_block_size_x));

    for (int band = 0; band < nbands; ++band)
        geoDataBlock[band]->fill(0);

    std::vector<std::unique_ptr<isce3::core::Matrix<T>>> geoDataBlockOffDiag;
    if (nbands_off_diag_terms > 0) {
        geoDataBlockOffDiag.reserve(nbands_off_diag_terms);
        for (int band = 0; band < nbands_off_diag_terms; ++band)
            geoDataBlockOffDiag.emplace_back(
                    std::make_unique<isce3::core::Matrix<T>>(
                            this_block_size_y, this_block_size_x));
        for (int band = 0; band < nbands_off_diag_terms; ++band)
            geoDataBlockOffDiag[band]->fill(0);
    }

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

    double a_min = radar_grid.sensingMid();
    double r_min = radar_grid.midRange();
    double a_max = radar_grid.sensingMid();
    double r_max = radar_grid.midRange();

    // pre-compute radar positions on the top of the geogrid
    bool flag_direction_line = true;

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
                          jj_0 + this_block_size_with_upsampling_x,
                          geogrid_upsampling, a11, r11, a_min, r_min, a_max,
                          r_max, a_last, r_last, dem_last, radar_grid, proj,
                          dem_interp_block, flag_direction_line);

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
                          jj_0 + this_block_size_with_upsampling_x,
                          geogrid_upsampling, a11, r11, a_min, r_min, a_max,
                          r_max, a_bottom, r_bottom, dem_bottom, radar_grid,
                          proj, dem_interp_block, flag_direction_line);

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

    _getRadarPositionVect(dem_x1, i_start, i_end, geogrid_upsampling, a11, r11,
                          a_min, r_min, a_max, r_max, a_left, r_left, dem_left,
                          radar_grid, proj, dem_interp_block,
                          flag_direction_line);

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

    _getRadarPositionVect(dem_x1, i_start, i_end, geogrid_upsampling, a11, r11,
                          a_min, r_min, a_max, r_max, a_right, r_right,
                          dem_right, radar_grid, proj, dem_interp_block,
                          flag_direction_line);

    // load radar grid data
    int offset_x = 0, offset_y = 0;
    int xbound = radar_grid.width() - 1;
    int ybound = radar_grid.length() - 1;

    std::vector<std::unique_ptr<isce3::core::Matrix<T2>>> rdrDataBlock;
    if (!is_radar_grid_single_block) {
        int margin_pixels = 25;
        offset_y = std::max((int) std::floor((a_min - start) / pixazm) -
                                    margin_pixels,
                            (int) 0);
        offset_x = std::max((int) std::floor((r_min - r0) / dr) - margin_pixels,
                            (int) 0);
        ybound = std::min((int) std::ceil((a_max - start) / pixazm) +
                                  margin_pixels,
                          (int) input_raster.length() - 1);
        int grid_size_y = ybound - offset_y;
        xbound = std::min((int) std::ceil((r_max - r0) / dr) + margin_pixels,
                          (int) (input_raster.width() - 1) *
                                  radar_grid_range_upsampling);
        int grid_size_x = xbound - offset_x;
        isce3::product::RadarGridParameters radar_grid_block =
                radar_grid.offsetAndResize(offset_y, offset_x, grid_size_y,
                                           grid_size_x);

        _getUpsampledBlock<T, T2>(
                rdrDataBlock, input_raster, offset_x, offset_y,
                radar_grid_block.width(), radar_grid_block.length(),
                flag_upsample_radar_grid, geocode_memory_mode, info);
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

#pragma omp atomic
            numdone++;
            if (numdone % progress_block == 0)
#pragma omp critical
                printf("\rgeocode progress: %d%%",
                       (int) (numdone / progress_block)),
                        fflush(stdout);

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

            // update "last" arrays (from lower left vertex)
            if (!std::isnan(a10)) {
                a_last[j] = a10;
                r_last[j] = r10;
                dem_last[j] = dem10;
            }

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
                dem11 = {dem_x1, dem_y1,
                         dem_interp_block.interpolateXY(dem_x1, dem_y1)};
                int converged = isce3::geometry::geo2rdr(
                        proj->inverse(dem11), _ellipsoid, _orbit, _doppler, a11,
                        r11, radar_grid.wavelength(), radar_grid.lookSide(),
                        _threshold, _numiter, 1.0e-8);
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

            int margin = isce3::core::AREA_PROJECTION_RADAR_GRID_MARGIN;

            // define slant-range window
            const int y_min = std::floor((std::min(std::min(a00, a01),
                                                   std::min(a10, a11)) -
                                          start) /
                                         pixazm) -
                              1;
            if (y_min < -margin ||
                y_min > ybound + 1)
                continue;
            const int x_min = std::floor((std::min(std::min(r00, r01),
                                                   std::min(r10, r11)) -
                                          r0) /
                                         dr) -
                              1;
            if (x_min < -margin ||
                x_min > xbound + 1)
                continue;
            const int y_max = std::ceil((std::max(std::max(a00, a01),
                                                  std::max(a10, a11)) -
                                         start) /
                                        pixazm) +
                              1;
            if (y_max > ybound + 1 + margin || y_max < -1 || y_max < y_min)
                continue;
            const int x_max = std::ceil((std::max(std::max(r00, r01),
                                                  std::max(r10, r11)) -
                                         r0) /
                                        dr) +
                              1;
            if (x_max > xbound + 1 + margin || x_max < -1 || x_max < x_min)
                continue;

            if (std::isnan(a00) || std::isnan(a01) || std::isnan(a10) ||
                std::isnan(a11)) {
                continue;
            }

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

            isce3::core::Matrix<double> w_arr(size_y, size_x);
            w_arr.fill(0);
            double w_total = 0;
            int plane_orientation;
            if (radar_grid.lookSide() == isce3::core::LookSide::Left)
                plane_orientation = -1;
            else
                plane_orientation = 1;

            isce3::geometry::areaProjIntegrateSegment(
                    y00, y01, x00, x01, size_y, size_x, w_arr, w_total,
                    plane_orientation);
            isce3::geometry::areaProjIntegrateSegment(
                    y01, y11, x01, x11, size_y, size_x, w_arr, w_total,
                    plane_orientation);
            isce3::geometry::areaProjIntegrateSegment(
                    y11, y10, x11, x10, size_y, size_x, w_arr, w_total,
                    plane_orientation);
            isce3::geometry::areaProjIntegrateSegment(
                    y10, y00, x10, x00, size_y, size_x, w_arr, w_total,
                    plane_orientation);

            double nlooks = 0;
            float area_total = 0;
            std::vector<T_out> cumulative_sum(nbands, 0);
            std::vector<T> cumulative_sum_off_diag_terms(nbands_off_diag_terms,
                                                         0);

            // add all slant-range elements that contributes to the geogrid
            // pixel
            for (int yy = 0; yy < size_y; ++yy) {
                for (int xx = 0; xx < size_x; ++xx) {
                    double w = w_arr(yy, xx);
                    int y = yy + y_min;
                    int x = xx + x_min;
                    if (w == 0 || w * w_total < 0)
                        continue;
                    else if (y - offset_y < 0 || x - offset_x < 0 ||
                             y >= ybound || x >= xbound) {
                        nlooks = std::numeric_limits<double>::quiet_NaN();
                        break;
                    }
                    w = std::abs(w);
                    if (flag_apply_rtc) {
                        float rtc_value = rtc_area(y, x);
                        if (std::isnan(rtc_value) || rtc_value < rtc_min_value)
                            continue;
                        nlooks += w;
                        if (isce3::is_complex<T_out>())
                            rtc_value = std::sqrt(rtc_value);
                        area_total += rtc_value * w;
                        if (flag_apply_rtc)
                            w /= rtc_value;
                    } else {
                        nlooks += w;
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

            // save geo-edges
            if (out_geo_rdr != nullptr) {
                // if first (top) line, save top right
                if (i == 0) {
                    out_geo_rdr_a(i, j + 1) = (a01 - start) / pixazm;
                    out_geo_rdr_r(i, j + 1) = (r01 - r0) / dr;
                }

                // if first (top left) pixel, save top left pixel
                if (i == 0 && j == 0) {
                    out_geo_rdr_a(i, j) = (a00 - start) / pixazm;
                    out_geo_rdr_r(i, j) = (r00 - r0) / dr;
                }

                // if first (left) column, save lower left
                if (j == 0) {
                    out_geo_rdr_a(i + 1, j) = (a10 - start) / pixazm;
                    out_geo_rdr_r(i + 1, j) = (r10 - r0) / dr;
                }

                // save lower left pixel
                out_geo_rdr_a(i + 1, j + 1) = (a11 - start) / pixazm;
                out_geo_rdr_r(i + 1, j + 1) = (r11 - r0) / dr;
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

            // x, y positions are binned by integer quotient (floor)
            const int x = (int) j / geogrid_upsampling;
            const int y = (int) i / geogrid_upsampling;

            if (flag_apply_rtc) {
                area_total /= nlooks;
            } else {
                area_total = 1;
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

            // divide by total and save result in the output array
            for (int band = 0; band < nbands; ++band)
                geoDataBlock[band]->operator()(y, x) += ((T_out)(
                        (cumulative_sum[band]) * abs_cal_factor_effective /
                        (nlooks * geogrid_upsampling * geogrid_upsampling)));

            if (nbands_off_diag_terms > 0) {
                for (int band = 0; band < nbands_off_diag_terms; ++band) {
                    geoDataBlockOffDiag[band]->operator()(y, x) +=
                            ((T)((cumulative_sum_off_diag_terms[band]) *
                                 abs_cal_factor_effective /
                                 (nlooks * geogrid_upsampling *
                                  geogrid_upsampling)));
                }
            }
        }
    }
    for (int band = 0; band < nbands; ++band) {
        for (int i = 0; i < this_block_size_y; ++i) {
            for (int j = 0; j < this_block_size_x; ++j) {
                T_out geo_value = geoDataBlock[band]->operator()(i, j);

                // no data
                if (std::abs(geo_value) == 0)
                    geoDataBlock[band]->operator()(i, j) = nan_t_out;

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
#pragma omp critical
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
                    /* 
                    Since std::numeric_limits<T_out>::quiet_NaN() with
                    complex T_out is (or may be) undefined, we take the "real type"
                    if T_out (i.e. float or double) to create the NaN value and
                    multiply it by the current pixel so that the output will be
                    real or complex depending on T_out and will contain NaNs.
                    */
                    using T_real = typename isce3::real<T>::type;
                    T geo_value_off_diag =
                            geoDataBlockOffDiag[band]->operator()(i, j);

                    // no data (complex)
                    if (std::abs(geo_value_off_diag) == 0)
                        geoDataBlockOffDiag[band]->operator()(i, j) =
                                std::numeric_limits<T_real>::quiet_NaN() *
                                geo_value_off_diag;

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

#pragma omp critical
            {
                out_off_diag_terms->setBlock(
                        geoDataBlockOffDiag[band]->data(),
                        block_x * block_size_x, block_y * block_size_y,
                        this_block_size_x, this_block_size_y, band + 1);
            }
        }
    }

    geoDataBlockOffDiag.clear();

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
                              block_y * block_size_y, this_block_size_x,
                              this_block_size_y, 1);
    }
}

template class Geocode<float>;
template class Geocode<double>;
template class Geocode<std::complex<float>>;
template class Geocode<std::complex<double>>;

// template <typename T>
std::vector<float> getGeoAreaElementMean(
        const std::vector<double>& x_vect, const std::vector<double>& y_vect,
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& input_dop,
        isce3::io::Raster& input_raster, isce3::io::Raster& dem_raster,
        bool flag_apply_rtc,
        isce3::geometry::rtcInputTerrainRadiometry input_terrain_radiometry, 
        isce3::geometry::rtcOutputTerrainRadiometry output_terrain_radiometry,  
        int exponent, geocodeOutputMode output_mode, double geogrid_upsampling,
        float rtc_min_value_db, double abs_cal_factor, float radar_grid_nlooks,
        float* out_nlooks, isce3::core::dataInterpMethod dem_interp_method,
        double threshold, int num_iter, double delta_range)
{

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

    std::string input_terrain_radiometry_str =
            get_input_terrain_radiometry_str(input_terrain_radiometry);

    info << "input radiometry: " << input_terrain_radiometry_str
         << pyre::journal::newline << "look side: " << radar_grid.lookSide()
         << pyre::journal::newline
         << "radar_grid length: " << radar_grid.length()
         << ", width: " << radar_grid.width() << pyre::journal::newline
         << "RTC min value [dB]: " << rtc_min_value_db << pyre::journal::endl;

    const double margin_x = std::abs(dx) * 10;
    const double margin_y = std::abs(dy) * 10;
    isce3::geometry::DEMInterpolator dem_interp;

    dem_interp.loadDEM(dem_raster, x0 - margin_x, xf + margin_x,
                       std::min(y0, yf) - margin_y,
                       std::max(y0, yf) + margin_y);

    isce3::core::Ellipsoid ellipsoid =
            isce3::core::Ellipsoid(isce3::core::EarthSemiMajorAxis,
                                   isce3::core::EarthEccentricitySquared);

    int epsg = dem_raster.getEPSG();
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(epsg));
    const double pixazm = radar_grid.azimuthTimeInterval();
    const double start = radar_grid.sensingStart() - 0.5 * pixazm;
    const double dr = radar_grid.rangePixelSpacing();
    double r0 = radar_grid.startingRange() - 0.5 * dr;

    double a = radar_grid.sensingMid();
    double r = radar_grid.midRange();

    if (x_vect.size() != y_vect.size()) {
        std::string error_message =
                "ERROR x and y vectors have a different number of elements.";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_message);
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
        int converged = isce3::geometry::geo2rdr(
                proj->inverse(dem11), ellipsoid, orbit, input_dop, a, r,
                radar_grid.wavelength(), radar_grid.lookSide(), threshold,
                num_iter, delta_range);
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

    isce3::product::RadarGridParameters radar_grid_cropped =
            radar_grid.offsetAndResize(y_min, x_min, ysize, xsize);

    info << "cropped radar_grid length: " << radar_grid_cropped.length()
         << ", width: " << radar_grid_cropped.width() << pyre::journal::newline;

    if (output_mode == geocodeOutputMode::INTERP) {
        std::string error_msg = "invalid option";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    if (abs_cal_factor != 1)
        info << "absolute calibration factor: " << abs_cal_factor
             << pyre::journal::endl;

    if (flag_apply_rtc) {

        std::string input_terrain_radiometry_str =
                get_input_terrain_radiometry_str(input_terrain_radiometry);
        info << "input radiometry: " << input_terrain_radiometry_str
             << pyre::journal::endl;
    }

    isce3::core::Matrix<float> rtc_area;
    std::unique_ptr<isce3::io::Raster> rtc_raster_unique_ptr;
    if (flag_apply_rtc) {

        info << "computing RTC area factor..." << pyre::journal::endl;
        rtc_raster_unique_ptr = std::make_unique<isce3::io::Raster>(
                "/vsimem/dummy", radar_grid_cropped.width(),
                radar_grid_cropped.length(), 1, GDT_Float32, "ENVI");
        isce3::geometry::rtcAreaMode rtc_area_mode =
                isce3::geometry::rtcAreaMode::AREA_FACTOR;
        isce3::geometry::rtcAlgorithm rtc_algorithm =
                isce3::geometry::rtcAlgorithm::RTC_AREA_PROJECTION;

        isce3::geometry::rtcMemoryMode rtc_memory_mode =
                isce3::geometry::rtcMemoryMode::RTC_SINGLE_BLOCK;

        computeRtc(radar_grid_cropped, orbit, input_dop, dem_raster,
                *rtc_raster_unique_ptr.get(), input_terrain_radiometry,
                output_terrain_radiometry, rtc_area_mode, rtc_algorithm,
                geogrid_upsampling * 2, rtc_min_value_db, radar_grid_nlooks,
                nullptr, rtc_memory_mode, dem_interp_method, threshold, num_iter,
                delta_range);

        rtc_area.resize(radar_grid_cropped.length(),
                        radar_grid_cropped.width());

        rtc_raster_unique_ptr->getBlock(rtc_area.data(), 0, 0,
                                              radar_grid_cropped.width(),
                                              radar_grid_cropped.length(), 1);

        info << "... done (RTC) " << pyre::journal::endl;
    }

    double rtc_min_value = 0;
    if (!std::isnan(rtc_min_value_db) && flag_apply_rtc) {
        rtc_min_value = std::pow(10., (rtc_min_value_db / 10.));
        info << "RTC min. value: " << rtc_min_value_db
             << " [dB] = " << rtc_min_value << pyre::journal::endl;
    }

    GDALDataType input_dtype = input_raster.dtype();
    if (exponent == 0 && GDALDataTypeIsComplex(input_dtype))
        exponent = 2;

    if (input_raster.dtype() == GDT_Float32) {
        info << "dtype: GDT_Float32" << pyre::journal::endl;
        return _getGeoAreaElementMean<float>(
                r_vect, a_vect, x_min, y_min, flag_apply_rtc,
                rtc_area, radar_grid_cropped,
                input_raster, rtc_min_value, out_nlooks,
                abs_cal_factor, radar_grid_nlooks);
    } else if (input_raster.dtype() == GDT_CFloat32) {
        info << "dtype: GDT_CFloat32" << pyre::journal::endl;
        return _getGeoAreaElementMean<std::complex<float>>(
                r_vect, a_vect, x_min, y_min, flag_apply_rtc,
                rtc_area, radar_grid_cropped,
                input_raster, rtc_min_value, out_nlooks,
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
        int x_min, int y_min, bool flag_apply_rtc,
        isce3::core::Matrix<float>& rtc_area,
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& input_raster, 
        float rtc_min_value, float* out_nlooks, double abs_cal_factor,
        float radar_grid_nlooks)
{

    pyre::journal::info_t info("isce.geometry._getGeoAreaElementMean");

    // number of bands in the input raster
    const int nbands = input_raster.numBands();
    const int size_y = radar_grid.length();
    const int size_x = radar_grid.width();
    info << "nbands: " << nbands << pyre::journal::endl;

    std::vector<std::unique_ptr<isce3::core::Matrix<T>>> rdrDataBlock;
    rdrDataBlock.reserve(nbands);

    for (int band = 0; band < nbands; ++band) {
        if (nbands == 1)
            info << "loading slant-range image..." << pyre::journal::endl;
        else
            info << "loading slant-range band: " << band << pyre::journal::endl;
        rdrDataBlock.emplace_back(
                std::make_unique<isce3::core::Matrix<T>>(size_y, size_x));

        input_raster.getBlock(rdrDataBlock[band]->data(), x_min, y_min,
                              size_x, size_y, band + 1);
    }

    isce3::core::Matrix<double> w_arr(size_y, size_x);
    w_arr.fill(0);

    int plane_orientation;
    if (radar_grid.lookSide() == isce3::core::LookSide::Left)
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
        isce3::geometry::areaProjIntegrateSegment(
                y00 - y_min, y01 - y_min, x00 - x_min, x01 - x_min, size_y,
                size_x, w_arr, w_total, plane_orientation);
    }
    std::vector<float> cumulative_sum(nbands);
    double nlooks = 0;

    Geocode<T> geo_obj;
    for (int y = 0; y < size_y; ++y)
        for (int x = 0; x < size_x; ++x) {
            double w = w_arr(y, x);
            if (w == 0)
                continue;
            if (flag_apply_rtc) {
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
                            rdrDataBlock[band]->operator()(y, x), w);
        }

    info << "nlooks: " << radar_grid_nlooks * std::abs(nlooks)
         << pyre::journal::endl;
    for (int band = 0; band < nbands; ++band) {
        cumulative_sum[band] = (cumulative_sum[band] * abs_cal_factor / nlooks);
        info << "mean value (band = " << band << "): " << cumulative_sum[band]
             << pyre::journal::endl;
    }

    if (out_nlooks != nullptr) {
        *out_nlooks = radar_grid_nlooks * std::abs(nlooks);
    }

    return cumulative_sum;
}

}} // namespace isce3::geocode
