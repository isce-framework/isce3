#include "GeocodeCov.h"
#include "GeocodePolygon.h"
#include "GeocodeHelpers.h"

#include <limits>

#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Projections.h>
#include <isce3/core/blockProcessing.h>
#include <isce3/core/Utilities.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/RTC.h>
#include <isce3/geometry/geometry.h>
#include <isce3/core/TypeTraits.h>


namespace isce3 { namespace geocode {

template <class T>
GeocodePolygon<T>::GeocodePolygon(
        const std::vector<double>& x_vect, const std::vector<double>& y_vect,
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Orbit& orbit,
        const isce3::core::Ellipsoid& ellipsoid,
        const isce3::core::LUT2d<double>& input_dop,
        isce3::io::Raster& dem_raster,
        double threshold, int num_iter, double delta_range) {

    pyre::journal::info_t _info("isce.geometry.GeocodePolygon");

    if (x_vect.size() != y_vect.size()) {
        std::string error_msg = "ERROR number of X- and Y-coordinates"
                                " do not match: " +
                                std::to_string(x_vect.size()) +
                                " != " + std::to_string(y_vect.size());
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
    if (x_vect.size() < 3) {
        std::string error_msg = "ERROR the polygon must have at least 3 vertices";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
    
    const auto minmax_x = \
        std::minmax_element(x_vect.begin(), x_vect.end());
    const double x0 = *minmax_x.first;
    const double xf = *minmax_x.second;

    const auto minmax_y = \
        std::minmax_element(y_vect.begin(), y_vect.end());
    const double y0 = *minmax_y.first;
    const double yf = *minmax_y.second;

    const double dx = dem_raster.dx();
    const double dy = dem_raster.dy();

    const double margin_x = std::abs(dx) * 10;
    const double margin_y = std::abs(dy) * 10;
    isce3::geometry::DEMInterpolator dem_interp;

    dem_interp.loadDEM(dem_raster, x0 - margin_x, xf + margin_x,
                       std::min(y0, yf) - margin_y,
                       std::max(y0, yf) + margin_y);

    int epsg = dem_raster.getEPSG();
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(epsg));

    const double pixazm = radar_grid.azimuthTimeInterval();
    const double start = radar_grid.sensingStart() - 0.5 * pixazm;
    const double dr = radar_grid.rangePixelSpacing();
    double r0 = radar_grid.startingRange() - 0.5 * dr;

    double a = radar_grid.sensingMid();
    double r = radar_grid.midRange();

    int n_elements = x_vect.size();
    _az_time_vect.reserve(n_elements);
    _slant_range_vect.reserve(n_elements);

    _info << "polygon indices (a, r): ";

    for (int i = 0; i < n_elements; ++i) {

        const double x = x_vect[i];
        const double y = y_vect[i];

        const Vec3 dem11 = {x, y, dem_interp.interpolateXY(x, y)};
        int converged = isce3::geometry::geo2rdr(
                proj->inverse(dem11), ellipsoid, orbit, input_dop, a, r,
                radar_grid.wavelength(), radar_grid.lookSide(), threshold,
                num_iter, delta_range);
        if (!converged) {
            _info << "WARNING convergence not found for vertex (x, y): " << x
                 << ", " << y << pyre::journal::endl;
            continue;
        }
        double a_index = (a - start) / pixazm;
        double r_index = (r - r0) / dr;

        _info << "(" << a_index << ", " << r_index << "), ";

        _az_time_vect.emplace_back(a_index);
        _slant_range_vect.emplace_back(r_index);
    }

    const auto minmax_r = \
        std::minmax_element(_slant_range_vect.begin(), 
                            _slant_range_vect.end());
    const double slant_range_min = *minmax_r.first;
    const double slant_range_max = *minmax_r.second;

    const auto minmax_a = \
        std::minmax_element(_az_time_vect.begin(), 
                            _az_time_vect.end());
    const double az_time_min = *minmax_a.first;
    const double az_time_max = *minmax_a.second;

    int margin = 10;
    _yoff = std::max(0, (int) std::floor(az_time_min) - margin);
    _xoff = std::max(0, (int) std::floor(slant_range_min) - margin);
    double yend = std::min((int) radar_grid.length() - 1, 
                           (int) std::ceil(az_time_max)) + margin;
    double xend = std::min((int) radar_grid.width() - 1, 
                           (int) std::ceil(slant_range_max)) + margin;
    _ysize = yend - _yoff;
    _xsize = xend - _xoff;
    _out_nlooks = 0;
    
    _threshold = threshold;
    _num_iter = num_iter;
    _delta_range = delta_range;
    _orbit = orbit;
    _ValidatePolygon(radar_grid);
}

template <class T>
void GeocodePolygon<T>::getPolygonMean(
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::LUT2d<double>& input_dop,
        isce3::io::Raster& input_raster,
        isce3::io::Raster& output_raster,
        isce3::io::Raster& dem_raster,
        bool flag_apply_rtc,
        isce3::geometry::rtcInputTerrainRadiometry input_terrain_radiometry, 
        isce3::geometry::rtcOutputTerrainRadiometry output_terrain_radiometry, 
        int exponent, double geogrid_upsampling,
        float rtc_min_value_db, double abs_cal_factor, float radar_grid_nlooks,
        isce3::io::Raster* output_off_diag_terms,
        isce3::io::Raster* output_radargrid_data, isce3::io::Raster* output_rtc,
        isce3::io::Raster* output_weights, isce3::core::dataInterpMethod interp_method) {
        
    pyre::journal::info_t _info("isce.geometry.getPolygonMean");

    _ValidatePolygon(radar_grid);
    if (std::isnan(geogrid_upsampling))
        geogrid_upsampling = 2;
    assert(geogrid_upsampling > 0);

    _info <<"look side: " << radar_grid.lookSide()
         << pyre::journal::newline
         << "radar_grid length: " << radar_grid.length()
         << ", width: " << radar_grid.width() << pyre::journal::newline
         << "RTC min value [dB]: " << rtc_min_value_db << pyre::journal::endl;

    _info << "cropping radar grid from index (a0: " << _yoff;
    _info << ", r0: " << _xoff << ") to index (af: " << _yoff + _ysize;
    _info << ", rf: " << _xoff + _xsize << ")" << pyre::journal::endl; 

    isce3::product::RadarGridParameters radar_grid_cropped =
            radar_grid.offsetAndResize(_yoff, _xoff, _ysize, _xsize);

    _info << "cropped radar_grid length: " << radar_grid_cropped.length()
            << ", width: " << radar_grid_cropped.width() << pyre::journal::newline;

    if (abs_cal_factor != 1)
        _info << "absolute calibration factor: " << abs_cal_factor
            << pyre::journal::endl;

    if (flag_apply_rtc) {

        std::string input_terrain_radiometry_str =
                get_input_terrain_radiometry_str(input_terrain_radiometry);
        _info << "input radiometry: " << input_terrain_radiometry_str
             << pyre::journal::endl;

        std::string output_terrain_radiometry_str =
                get_output_terrain_radiometry_str(output_terrain_radiometry);
        _info << "output radiometry: " << output_terrain_radiometry_str
             << pyre::journal::endl;
    }

    isce3::core::Matrix<float> rtc_area;
    std::unique_ptr<isce3::io::Raster> rtc_raster_unique_ptr;
    if (flag_apply_rtc) {

        isce3::io::Raster* rtc_raster;
        // if RTC (area factor) raster does not needed to be saved,
        // initialize it as a GDAL memory virtual file
        if (output_rtc == nullptr) {
            std::string vsimem_ref = (
                "/vsimem/" + getTempString("geocode_polygon_rtc"));
            rtc_raster_unique_ptr = std::make_unique<isce3::io::Raster>(
                    vsimem_ref, radar_grid_cropped.width(),
                    radar_grid_cropped.length(), 1, GDT_Float32, "ENVI");
            rtc_raster = rtc_raster_unique_ptr.get();
        }

        // Otherwise, copies the pointer to the output RTC file
        else
            rtc_raster = output_rtc;

        _info << "computing RTC area factor..." << pyre::journal::endl;

        isce3::geometry::rtcAreaMode rtc_area_mode =
                isce3::geometry::rtcAreaMode::AREA_FACTOR;
        isce3::geometry::rtcAlgorithm rtc_algorithm =
                isce3::geometry::rtcAlgorithm::RTC_AREA_PROJECTION;
        isce3::geometry::rtcAreaBetaMode rtc_area_beta_mode =
                isce3::geometry::rtcAreaBetaMode::AUTO;

        isce3::core::MemoryModeBlocksY rtc_memory_mode =
                isce3::core::MemoryModeBlocksY::SingleBlockY;

        isce3::io::Raster* out_sigma = nullptr;

        computeRtc(radar_grid_cropped, _orbit, input_dop, dem_raster,
                   *rtc_raster, input_terrain_radiometry,
                   output_terrain_radiometry, rtc_area_mode,
                   rtc_algorithm, rtc_area_beta_mode,
                   geogrid_upsampling * 2, rtc_min_value_db,
                   out_sigma, rtc_memory_mode, interp_method, _threshold,
                   _num_iter, _delta_range);

        rtc_area.resize(radar_grid_cropped.length(),
                        radar_grid_cropped.width());

        rtc_raster->getBlock(rtc_area.data(), 0, 0, radar_grid_cropped.width(),
                             radar_grid_cropped.length(), 1);

        _info << "... done (RTC) " << pyre::journal::endl;
    }

    double rtc_min_value = 0;
    if (!std::isnan(rtc_min_value_db) && flag_apply_rtc) {
        rtc_min_value = std::pow(10., (rtc_min_value_db / 10.));
        _info << "RTC min. value: " << rtc_min_value_db
              << " [dB] = " << rtc_min_value << pyre::journal::endl;
    }

    _info <<"geogrid upsampling: " << geogrid_upsampling
         << pyre::journal::newline;

    GDALDataType input_dtype = input_raster.dtype();
    if (exponent == 0 && GDALDataTypeIsComplex(input_dtype))
        exponent = 2;

    if (input_raster.dtype() == GDT_Float32) {
        _info << "input dtype: GDT_Float32" << pyre::journal::endl;
        _info << "output dtype: GDT_Float32" << pyre::journal::endl;
        _getPolygonMean<float>(
                rtc_area, radar_grid_cropped, input_raster, output_raster,
                flag_apply_rtc, rtc_min_value, abs_cal_factor, radar_grid_nlooks,
                output_off_diag_terms, output_radargrid_data, output_weights);
    } else if (input_raster.dtype() == GDT_CFloat32 && exponent == 2) {
        _info << "input dtype: GDT_CFloat32" << pyre::journal::endl;
        _info << "output dtype: GDT_Float32" << pyre::journal::endl;
        _getPolygonMean<float>(
                rtc_area, radar_grid_cropped, input_raster, output_raster,
                flag_apply_rtc, rtc_min_value, abs_cal_factor, radar_grid_nlooks,
                output_off_diag_terms, output_radargrid_data, output_weights);
    } else if (input_raster.dtype() == GDT_CFloat32 && exponent == 1) {
        _info << "input dtype: GDT_CFloat32" << pyre::journal::endl;
        _info << "output dtype: GDT_CFloat32" << pyre::journal::endl;
        _getPolygonMean<std::complex<float>>(
                rtc_area, radar_grid_cropped, input_raster, output_raster,
                flag_apply_rtc, rtc_min_value, abs_cal_factor, radar_grid_nlooks,
                output_off_diag_terms, output_radargrid_data, output_weights);
    } else
        _info << "ERROR not implemented for datatype: " << input_raster.dtype()
             << pyre::journal::endl;

}

template<class T>
template<class T_out>
void GeocodePolygon<T>::_getPolygonMean(
        isce3::core::Matrix<float>& rtc_area,
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& input_raster,
        isce3::io::Raster& output_raster,
        bool flag_apply_rtc,
        float rtc_min_value,
        double abs_cal_factor, float radar_grid_nlooks,
        isce3::io::Raster* output_off_diag_terms,
        isce3::io::Raster* output_radargrid_data,
        isce3::io::Raster* output_weights) {

    pyre::journal::info_t _info("isce.geometry._getPolygonMean");

    using isce3::math::complex_operations::operator*;

    // number of bands in the input raster
    const int nbands = input_raster.numBands();
    _info << "nbands: " << nbands << pyre::journal::endl;

    int nbands_off_diag_terms = 0;
    if (output_off_diag_terms != nullptr) {
        _info << "nbands (diagonal terms): " << nbands << pyre::journal::endl;
        nbands_off_diag_terms = nbands*(nbands - 1)/2; 
        _info << "nbands (off-diagonal terms): " << nbands_off_diag_terms 
             << pyre::journal::endl;
        assert(output_off_diag_terms->numBands() == nbands_off_diag_terms);
        _info << "full covariance: true" << pyre::journal::endl;
        if (!GDALDataTypeIsComplex(input_raster.dtype())){
            std::string error_msg = "Input raster must be complex to"
                                    " generate full-covariance matrix";
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
        }
        if (!GDALDataTypeIsComplex(output_off_diag_terms->dtype())){
            std::string error_msg = "Off-diagonal raster must be complex";
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
        }
    } else {
        _info << "nbands: " << nbands << pyre::journal::endl;
        _info << "full covariance: false" << pyre::journal::endl;
    }

    std::vector<std::unique_ptr<isce3::core::Matrix<T>>> rdrDataBlock;
    rdrDataBlock.reserve(nbands);

    for (int band = 0; band < nbands; ++band) {
        if (nbands == 1)
            _info << "loading slant-range image..." << pyre::journal::endl;
        else
            _info << "loading slant-range band: " << band << pyre::journal::endl;
        rdrDataBlock.emplace_back(
                std::make_unique<isce3::core::Matrix<T>>(_ysize, _xsize));

        input_raster.getBlock(rdrDataBlock[band].get()->data(), _xoff, _yoff,
                              _xsize, _ysize, band + 1);
    }

    isce3::core::Matrix<double> w_arr(_ysize, _xsize);
    w_arr.fill(0);

    int plane_orientation;
    if (radar_grid.lookSide() == isce3::core::LookSide::Left)
        plane_orientation = -1;
    else
        plane_orientation = 1;

    double w_total = 0;
    for (int i = 0; i < _slant_range_vect.size(); ++i) {

        double x00, y00, x01, y01;
        y00 = _az_time_vect[i];
        x00 = _slant_range_vect[i];

        if (i < _slant_range_vect.size() - 1) {
            y01 = _az_time_vect[i + 1];
            x01 = _slant_range_vect[i + 1];
        } else {
            y01 = _az_time_vect[0];
            x01 = _slant_range_vect[0];
        }
        isce3::geometry::areaProjIntegrateSegment(y00 - _yoff, y01 - _yoff, x00 - _xoff,
                                 x01 - _xoff, _ysize, _xsize, w_arr, w_total,
                                 plane_orientation);
    }
    std::vector<T_out> cumulative_sum(nbands);
    std::vector<T> cumulative_sum_off_diag_terms(nbands_off_diag_terms, 0);
    double nlooks = 0;

    isce3::core::Matrix<T_out> output_radargrid_data_array;
    if (output_radargrid_data != nullptr) {
        output_radargrid_data_array.resize(_ysize, _xsize);
        output_radargrid_data_array.fill(std::numeric_limits<T_out>::quiet_NaN());
    }

    Geocode<T> geo_obj;
    for (int y = 0; y < _ysize; ++y)
        for (int x = 0; x < _xsize; ++x) {
            double w = w_arr(y, x);
            if (w == 0)
                continue;
            /*  
            condition masks out layover:
            if (w * w_total < 0) {
                w_arr(y, x) = 0;
                continue;
            } 
            */
            w = std::abs(w);
            if (flag_apply_rtc) {
                const float rtc_value = rtc_area(y, x);
                if (std::isnan(rtc_value) || rtc_value < rtc_min_value)
                    continue;
                nlooks += w;
                w /= rtc_value;
            } else {
                nlooks += w;
            }
        
            int band_index = 0;
            for (int band_1 = 0; band_1 < nbands; ++band_1) {
                T v1 = rdrDataBlock[band_1].get()->operator()(y, x);
                _accumulate(cumulative_sum[band_1], v1, w);
                if (output_radargrid_data != nullptr) {
                    T_out out_radar;
                    _convertToOutputType(v1, out_radar);
                    output_radargrid_data_array(y, x) = out_radar * std::abs(w);
                }

                if (nbands_off_diag_terms > 0) {
                    for (int band_2 = 0; band_2 < nbands; ++band_2) {
                        if (band_2 <= band_1)
                            continue;
                        _accumulate(cumulative_sum_off_diag_terms[band_index],
                            v1 * std::conj(rdrDataBlock[band_2].get()->operator()(y, x)), 
                            w);
                        band_index++;
                    }
                }
            }
        }

    if (output_radargrid_data != nullptr)
        output_radargrid_data->setBlock(output_radargrid_data_array.data(), 0,
                                        0, _xsize, _ysize, 1);

    if (output_weights != nullptr)
        output_weights->setBlock(w_arr.data(), 0, 0, _xsize, _ysize, 1);

    _info << "nlooks: " << radar_grid_nlooks * std::abs(nlooks)
         << pyre::journal::endl;

    for (int band = 0; band < nbands; ++band) {
        cumulative_sum[band] *= abs_cal_factor / nlooks;
        _info << "mean value (band = " << band + 1 << "): " << cumulative_sum[band]
             << pyre::journal::endl;
    }
    output_raster.setBlock(cumulative_sum, 0, 0, nbands, 1);
    if (nbands_off_diag_terms > 0) {
        for (int band = 0; band < nbands_off_diag_terms; ++band) {
            cumulative_sum_off_diag_terms[band] *= abs_cal_factor / nlooks;
            _info << "mean value (off diag band = " << band + 1
                  << "): " << cumulative_sum_off_diag_terms[band]
                  << pyre::journal::endl;
        }
        output_off_diag_terms->setBlock(cumulative_sum_off_diag_terms, 0, 0,
                                        nbands_off_diag_terms, 1);
    }

    _out_nlooks = radar_grid_nlooks * std::abs(nlooks);

}

template<class T>
void GeocodePolygon<T>::_ValidatePolygon(
        const isce3::product::RadarGridParameters& radar_grid)
{
    if (_xoff >= radar_grid.width()) {
        std::string error_msg = "ERROR start X is greater than or equal to the "
                                "radargrid width: " +
                                std::to_string(_xoff) +
                                " >= " + std::to_string(_xsize);
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), error_msg);  
    }

    if (_yoff >= radar_grid.length()) {
        std::string error_msg = "ERROR start Y is greater than or equal to the "
                                "radargrid length: " +
                                std::to_string(_xoff) +
                                " >= " + std::to_string(_xsize);
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), error_msg);  
    }

    if (_xoff + _xsize <= 0) {
        std::string error_msg = "ERROR end X is less than or equal to the "
                                "first radargrid X index: " +
                                std::to_string(_xoff) + " <= 0";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), error_msg);  
    }

    if (_yoff + _ysize <= 0) {
        std::string error_msg = "ERROR start Y is less than or equal to the "
                                "first radargrid Y index: " +
                                std::to_string(_xoff) + " <= 0";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), error_msg);  
    }

    if (_ysize < 0) {
        std::string error_msg = "invalid Y-size: " + std::to_string(_ysize);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
    if (_xsize < 0) {
        std::string error_msg = "invalid X-size: " + std::to_string(_xsize);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
}

template class GeocodePolygon<float>;
template class GeocodePolygon<double>;
template class GeocodePolygon<std::complex<float>>;
template class GeocodePolygon<std::complex<double>>;

} // namespace geocode
} // namespace isce3
