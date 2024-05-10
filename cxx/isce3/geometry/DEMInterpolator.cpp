// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

#include <cmath>
#include "DEMInterpolator.h"

#include <isce3/core/Projections.h>
#include <isce3/io/Raster.h>

/** Set EPSG code for input DEM */
void isce3::geometry::DEMInterpolator::epsgCode(int epsgcode) {
    _epsgcode = epsgcode;
    _proj = isce3::core::makeProjection(epsgcode);
}

// Load DEM subset into memory
/** @param[in] demRaster input DEM raster to subset
 * @param[in] min_x Easting/Longitude of western edge of bounding box
 * @param[in] max_x Easting/Longitude of eastern edge of bounding box
 * @param[in] min_y Northing/Latitude of southern edge of bounding box
 * @param[in] max_y Northing/Latitude of northern edge of bounding box
 * @returns Error code
 *
 * Loads a DEM subset given the extents of a bounding box */
isce3::error::ErrorCode isce3::geometry::DEMInterpolator::loadDEM(
        isce3::io::Raster& demRaster, double min_x, double max_x, double min_y,
        double max_y, const int dem_raster_band)
{

    // Initialize journal
    pyre::journal::warning_t warning("isce.core.Geometry");

    // Get original GeoTransform using raster
    double geotransform[6];
    demRaster.getGeoTransform(geotransform);
    const double delta_y = geotransform[5];
    const double delta_x = geotransform[1];

    if (delta_x < 0) {
        warning << "The DEM pixel spacing in the X-/longitude direction"
                << " must be positive. Encountered value: "
                << delta_x << pyre::journal::endl;
        return isce3::error::ErrorCode::InvalidDem;
    }

    // Compute starting coordinate at pixels edge
    const double dem_y0 = geotransform[3];
    double dem_x0 = geotransform[0];

    //Initialize projection
    int epsgcode = demRaster.getEPSG();
    _epsgcode = epsgcode;
    _proj = isce3::core::makeProjection(epsgcode);

    /* If DEM in geographic coordinates (i.e. EPSG is 4326),
       we need to check for DEM file discontinuity (DFD) around dateline
       crossing (lon +/-180) or Greenwich Meridian crossing (lon 0 or 360).

        X positions as longitude (EPSG 4326) would be ideally given in
    either of two range conventions:
        1 - [-180, +180];
        2 - [0: +360].

        In practice however, datasets may be given in longitude ranges
    such as, for example, [-180.25, +179.75] or [-0.25, +359.75]. Also,
    if there is dateline crossing from the user coordinates `min_x` and
    `max_x`, `max_x` may "cross" the +180 value even if convention 1 is used.

        We want to make sure that the user-provided and DEM longitudes
    are comparable. We first "wrap" min_x and max_x by subtracting an integer
    multiple of 360 from each such that max_x falls within the closed interval
    [-180 - delta_x, 360 + delta_x] and the relative difference between min_x
    and max_x is preserved. Then, if max_x is less than min_x due to wrap-around,
    we add 360 to max_x to ensure that it's greater than min_x.

    A similar transformation is applied to the DEM extents to normalize them
    to the same range.

    */

    if (epsgcode == 4326) {

        /* For DEM in EPSG 4326 (lat/lon) max longitude range is
        360 + 2 DEM pixels */
        if (max_x - min_x > 360 + 2 * delta_x) {
            double new_max_x = min_x + 360 + delta_x;
            warning << "Longitude range from " << min_x
                    << " to " << max_x << " degrees exceed maximum"
                    << " longitude range of 360 degrees. The"
                    << " eastern boundary will be updated to "
                    << new_max_x << pyre::journal::endl;
            max_x = new_max_x;
        }

        /* Wrap equally `min_x` and `max_x` so that `max_x` is within
        longitudes [-180 - delta_x, 360 + delta_x] */
        if (min_x < -180 - delta_x || max_x < -180 - delta_x ||
                max_x > 360 + delta_x) {
            double n_wraps = std::floor(max_x / 360);
            min_x -= n_wraps * 360;
            max_x -= n_wraps * 360;
        }
        // make sure that both coordinates are greater than -180 - delta_x
        if (min_x < -180 - delta_x || max_x < -180 - delta_x) {
            min_x += 360;
            max_x += 360;
        }

        /* If `min_x` and `max_x` positions are given in convention 1 and
        these points cross the dateline (longitude = +/- 180), we wrap
        up `max_x` so that it will greater than `min_x`.
        */
        if (min_x > 0 && max_x < 0) {
            max_x += 360;
        }

        // Wrap `dem_x0` to longitude range [-180 - delta_x, 360 + delta_x]
        if (dem_x0 > 360 + delta_x || dem_x0 < -180 - delta_x) {
            dem_x0 = std::fmod(dem_x0, 360);
        }
    }

    // Compute ending coordinate at pixels edge
    const double dem_yf = dem_y0 + demRaster.length() * delta_y;
    const double dem_xf = dem_x0 + demRaster.width() * delta_x;

    /*
        Next, we make sure that the user-provided and DEM longitudes
    are comparable. Using the DEM longitudes as reference, we check if we
    should "wrap" user-provided positions `min_x` and `max_x` to match DEM
    bounds `dem_x0` and `dem_xf`, by adding 360 or substracting 360 degrees.

        A: Wrap user-provided positions "up" by adding 360:
        B: Wrap user-provided positions "down" by subtracting 360.

    ------------------------------------------------------------------
    Condition A:
    ------------------------------------------------------------------

        Instead of using conventions 1 and 2 above, we can define the valid
    longitude domain using the DEM eastern edge:

        [`dem_xf` - 360, `dem_xf`]

        Notice that for testing condition A, we just need to check if the
    user-provided western bound `min_x` is smaller than `dem_xf` -360.
    If so, we wrap `min_x` and `max_x` "up" by adding 360 to both coordinates.

        For example, a user-provide longitude of -10 will only need to be "wrapped
    up" (by adding 360) if the DEM last position is greater than 350, i.e.,
    -10 + 360. If the DEM last position is 340, we could consider that
    the valid domain is [-20: 340], i.e. [340 - 360: 340], and nothing needs
    to be done.

        To account for one extra pixel in the edges, we can consider the
    domain as:

        [lon_domain_min, lon_domain_max] = [
            `dem_xf` - 360 - delta_x, `dem_xf` + delta_x]

    Condition A becomes:
        - min_x < dem_xf - 360 - delta_x

    ------------------------------------------------------------------
    Condition B:
    ------------------------------------------------------------------

        For condition B, we can use a similar checking as the one used for
    condition A, i.e., user-defined eastern bound `max_x` is compared to
    DEM western bound `dem_x0`. However, the wrapping is only necessary
    if there's no DEM file discontinuity between `min_x` and `max_x`.

        Fox example, if the DEM is defined from [-180, 180] and the user
    bounds are defined from 170 to 190, there's no need to wrap coordinates
    170 and 190 down to the range [-180, 180]. However, if the user-defined
    bounds are from 185 to 190, we want to wrap these values down to -175 and
    -170, respectively, so that the domain matches the DEM domain.

    Condition B becomes:
        - There's no DEM file discontinuity between `min_x` and `max_x`;
        - `max_x` > dem_x0` + 360 + delta_x

    These tests can be simplified as:
        - min_x > 180 (no DEM file discontinuity and user positions are at the
        eastern side of the DEM);
        - max_x > dem_x0 + 360 + delta_x

    */

    if (epsgcode == 4326 && min_x < dem_xf - 360 - delta_x) {
        // 360 should be added to min_x and max_x
        min_x += 360;
        max_x += 360;
    }
    else if (epsgcode == 4326 && min_x > 180 &&
             max_x > dem_x0 + 360 + delta_x) {
        // 360 should be subtracted from min_x and max_x
        min_x -= 360;
        max_x -= 360;
    }

    /*
   ==================================================================
    DEM file discontinuity - Dateline crossing (example 1):

                                  min_x           max_x
                                     *-------------*
   -180 deg                             +180 deg
       *------------------------------------*
    dem_x0                               dem_xf
                                    (~ dem_x0 + 360)

    ==================================================================
    DEM file discontinuity - Greenwich Meridian crossing (example 2):

                                  min_x           max_x
                                     *-------------*
     0 deg                              +360 deg
       *------------------------------------*
    dem_x0                               dem_xf
                                    (~ dem_x0 + 360)

    ==================================================================
    No DEM file discontinuity (example 1):

                 min_x           max_x
                    *-------------*
       *------------------------------------*
    dem_x0                               dem_xf

    ==================================================================
    No DEM file discontinuity (example 2):

                                   min_x           max_x
                                      *-------------*
       *------------------------------------*                    *
    dem_x0                               dem_xf           (~ dem_x0 + 360)

    ==================================================================

    Check for DEM file discontinuity and DEM having data at both
    sides of the discontinuity:

    1. If DEM in geographic coordinates;
    2. If dem_xf (from the DEM) is between min_x and max_x
    3. dem_x0 + 360 is between min_x and max_x

    Notice that conditions 1-3 will only be satisfied if min_x and max_x
    cross the DEM file discontinuity

    If there is no DEM file discontinuity, i.e.,
    `flag_dem_file_discontinuity = false`, we read a single block from
    the DEM. Otherwise, we divide the DEM in two blocks, i.e. at the left and
    at the right side of the discontinuity.
    */

    bool flag_dem_file_discontinuity = (epsgcode == 4326) &&
            (min_x <= dem_xf) && (dem_xf < max_x) &&
            (min_x <= dem_x0 + 360) && (dem_x0 + 360 < max_x);

    // Validate requested geographic bounds with input DEM raster
    if (min_x < dem_x0) {
        warning << "West limit may be insufficient for global height range"
                << pyre::journal::endl;
        min_x = dem_x0;
    }
    if (max_x > dem_xf && !flag_dem_file_discontinuity) {
        warning << "East limit may be insufficient for global height range"
                << pyre::journal::endl;
        max_x = dem_xf;
    }
    if (min_y < dem_yf) {
        warning << "South limit may be insufficient for global height range"
                << pyre::journal::endl;
        min_y = dem_yf;
    }
    if (max_y > dem_y0) {
        warning << "North limit may be insufficient for global height range"
                << pyre::journal::endl;
        max_y = dem_y0;
    }

    // Compute pixel coordinates for geographic bounds (wrt edges)
    auto min_x_idx = static_cast<long>(std::floor((min_x - dem_x0) / delta_x));
    auto max_x_idx = static_cast<long>(std::ceil((max_x - dem_x0) / delta_x));
    auto min_y_idx = static_cast<long>(std::floor((max_y - dem_y0) / delta_y));
    auto max_y_idx = static_cast<long>(std::ceil((min_y - dem_y0) / delta_y));

    // Store actual starting lat/lon for raster subset (wrt pixels' center)
    _xstart = dem_x0 + (0.5 + min_x_idx) * delta_x;
    _ystart = dem_y0 + (0.5 + min_y_idx) * delta_y;
    _deltax = delta_x;
    _deltay = delta_y;

    // Get DEMInterpolator width and length
    long width = max_x_idx - min_x_idx;
    long length = max_y_idx - min_y_idx;

    // Make sure raster subset is does not extrapolate raster dimensions
    if (!flag_dem_file_discontinuity && min_x_idx + width > demRaster.width()) {
        width = demRaster.width() - min_x_idx;
        max_x_idx = min_x_idx + width;
    }
    if (!flag_dem_file_discontinuity && min_y_idx + length > demRaster.length()) {
        length = demRaster.length() - min_y_idx;
        max_y_idx = min_y_idx + length;
    }

    // If DEM has no valid points, escape
    if (width <= 0 || length <= 0) {
        warning << "The requested area is outside DEM limits"
                << pyre::journal::endl;
        return isce3::error::ErrorCode::OutOfBoundsDem;
    }

    // Resize DEM array
    _dem.resize(length, width);

    if (!flag_dem_file_discontinuity) {
        // Read single block from DEM
        demRaster.getBlock(_dem.data(), min_x_idx, min_y_idx, width, length,
                           dem_raster_band);

    } else {

        // Fill DEM array with NaN values
        _dem.fill(std::numeric_limits<float>::quiet_NaN());

        // Read DEM in two blocks "unrolling" the western side of the DEM around
        // the DEM file discontinuity
        const long width_discontinuity_left = demRaster.width() - min_x_idx;

        if (width_discontinuity_left > 0) {
            isce3::core::Matrix<float> dem_discontinuity_left;

            dem_discontinuity_left.resize(length, width_discontinuity_left);
            demRaster.getBlock(dem_discontinuity_left.data(), min_x_idx,
                               min_y_idx, width_discontinuity_left, length,
                               dem_raster_band);

            _Pragma("omp parallel for schedule(dynamic)")
            for (long i=0; i < length; ++i) {
                for (long j=0; j < width_discontinuity_left; ++j) {
                    _dem(i, j) = dem_discontinuity_left(i, j);
                }
            }
        }

        /*
        The W/E index (wrt DEM grid) of the first pixel in the
        right side of the DEM file discontinuity equals demRaster.width().

        The intuitive solution would be to place the first pixel
        (index 0) at the left side of the DEM. This is the most common
        case, however, there may be fewer or extra pixels in the border,
        therefore we need to locate the position of the first pixel at the
        right side of the DEM file discontinuity and wrap it to the left side
        of the DEM. In terms of index position, the wrapping is computed by
        subtracting 360.0 / delta_x from it's index which is demRaster.width().
        */
        const double wrapped_next_pixel_idx_ideal = (demRaster.width() -
                                                     360.0 / delta_x);

        /*
        Since the division of 360 by `delta_x` may not be an integer
        number, we need to round it to the closest integer to obtain the
        effective wrapped next pixel position
        */
        const long wrapped_next_pixel_idx = std::round(
            wrapped_next_pixel_idx_ideal);

        /*
        However, if `wrapped_next_pixel_idx_ideal` and `wrapped_next_pixel_idx`
        are too different, the array at the right side of the DEM file
        discontinuity will be shifted. We add a check to make sure that the
        shift is below a threshold (< 1e-5 of the pixel size).
        */
        const double position_diff_threshold = 1e-5;
        const double position_diff = std::abs(wrapped_next_pixel_idx_ideal -
                                              wrapped_next_pixel_idx);
        if (position_diff > position_diff_threshold) {
            warning << "DEM position differences at the left and right sides"
                    << " of the DEM file discontinuity exceed"
                    << " error threshold: " << position_diff << " (threshold: "
                    << position_diff_threshold << ")."
                    << pyre::journal::endl;
            return isce3::error::ErrorCode::InvalidDem;
        }

        // Take the idx of the first pixel to be loaded. It's usually 0.
        const long min_x_idx_discontinuity_right = std::max(
            static_cast<long>(0), wrapped_next_pixel_idx);

        // Take the position of the last pixel to be loaded:
        const long max_x_idx_discontinuity_right = std::min(
            static_cast<long>(demRaster.width() - 1),
            static_cast<long>(std::ceil((max_x - 360 - dem_x0) / delta_x)));

        // Compute the width of the block to be loaded
        const long width_discontinuity_right = (max_x_idx_discontinuity_right -
                                           min_x_idx_discontinuity_right);

        if (width_discontinuity_right > 1) {
            isce3::core::Matrix<float> dem_discontinuity_right;

            dem_discontinuity_right.resize(length, width_discontinuity_right);
            demRaster.getBlock(dem_discontinuity_right.data(),
                               min_x_idx_discontinuity_right, min_y_idx,
                               width_discontinuity_right, length,
                               dem_raster_band);

            _Pragma("omp parallel for schedule(dynamic)")
            for (long i=0; i < length; ++i) {
                for (long j=0; j < width_discontinuity_right; ++j) {
                    /*
                    Convert index position j from the right side of the DEM
                    file discontinuity to the left side (by adding
                    360.0 / delta_x) and subtract from the results the start
                    position of the DEM to get the index j wrt to DEM
                    array _dem.
                    */
                    const long dem_array_j = static_cast<long>(
                        (min_x_idx_discontinuity_right + j) + 360.0 / delta_x - min_x_idx);

                    if (dem_array_j > width - 1) {
                        continue;
                    }
                    _dem(i, dem_array_j) = dem_discontinuity_right(i, j);
                }
            }
        }
    }

    // Initialize internal interpolator
    _interp = std::unique_ptr<isce3::core::Interpolator<float>>(isce3::core::createInterpolator<float>(_interpMethod));

    // Indicate we have loaded a valid raster
    _haveRaster = true;

    // Since we just loaded the data we don't know the stats anymore.
    // NOTE No need for this flag if we change the design to always calc stats.
    _haveStats = false;

    return isce3::error::ErrorCode::Success;
}


// Load DEM into memory
/** @param[in] demRaster input DEM raster to subset
  *
  * Loads the entire DEM */
void isce3::geometry::DEMInterpolator::
loadDEM(isce3::io::Raster & demRaster, const int dem_raster_band) {

    //Get the dimensions of the raster
    int width = demRaster.width();
    int length = demRaster.length();


    // Get original GeoTransform using raster
    double geotransform[6];
    demRaster.getGeoTransform(geotransform);
    const double delta_y = geotransform[5];
    const double delta_x = geotransform[1];
    // Use center of pixel as starting coordinate
    const double firstY = geotransform[3] + 0.5 * delta_y;
    const double firstX = geotransform[0] + 0.5 * delta_x;

    //Initialize projection
    int epsgcode = demRaster.getEPSG();
    _epsgcode = epsgcode;
    _proj = isce3::core::makeProjection(epsgcode);

    // Store actual starting lat/lon for raster subset
    _xstart = firstX;
    _ystart = firstY;
    _deltax = delta_x;
    _deltay = delta_y;

    // Resize memory
    _dem.resize(length, width);

    // Read in the DEM
    demRaster.getBlock(_dem.data(), 0, 0, width, length, dem_raster_band);

    // Initialize internal interpolator
    _interp = std::unique_ptr<isce3::core::Interpolator<float>>(isce3::core::createInterpolator<float>(_interpMethod));

    // Indicate we have loaded a valid raster
    _haveRaster = true;

    // Since we just loaded the data we don't know the stats anymore.
    // NOTE No need for this flag if we change the design to always calc stats.
    _haveStats = false;
}


// Debugging output
void isce3::geometry::DEMInterpolator::
declare() const {
    pyre::journal::info_t info("isce.core.DEMInterpolator");
    info << "Actual DEM bounds used:" << pyre::journal::newline
         << "Top Left: " << _xstart << " " << _ystart << pyre::journal::newline
         << "Bottom Right: " << _xstart + _deltax * (_dem.width() - 1) << " "
         << _ystart + _deltay * (_dem.length() - 1) << " " << pyre::journal::newline
         << "Spacing: " << _deltax << " " << _deltay << pyre::journal::newline
         << "Dimensions: " << _dem.width() << " " << _dem.length() << pyre::journal::endl;
}

void isce3::geometry::DEMInterpolator::
computeMinMaxMeanHeight(float &minValue, float &maxValue, float &meanValue) {
    pyre::journal::info_t info("isce.core.DEMInterpolator.computeMinMaxMeanHeight");

    // Default to reference height
    minValue = _refHeight;
    maxValue = _refHeight;
    meanValue = _refHeight;

    // If a DEM raster exists, proceeed to computations
    if (_haveRaster and not _haveStats) {
        info << "Computing DEM statistics" << pyre::journal::newline;

        minValue = std::numeric_limits<float>::max();
        maxValue = -std::numeric_limits<float>::max();
        double sum = 0.0;
        auto n_valid = _dem.length() * _dem.width();
        // loop over all values in DEM raster
#pragma omp parallel for collapse(2) reduction(min : minValue)  \
                                     reduction(max : maxValue)  \
                                     reduction(+ : sum)         \
                                     reduction(- : n_valid)
        for (size_t i = 0; i < _dem.length(); ++i) {
            for (size_t j = 0; j < _dem.width(); ++j) {
                float value = _dem(i,j);

                // skip NaN and decrement denominator
                if (std::isnan(value)) {
                    n_valid--;
                    continue;
                }

                maxValue = std::max(value, maxValue);
                minValue = std::min(value, minValue);
                sum += value;
            }
        }
        meanValue = sum / n_valid;

        // Store updated statistics
        _haveStats = true;
        _minValue = minValue;
        _meanValue = meanValue;
        _maxValue = maxValue;

        // Update reference height so it's in bounds.
        refHeight(meanValue);

    } else if (_haveRaster) {
        info << "Using existing DEM statistics" << pyre::journal::newline;
        minValue = _minValue;
        meanValue = _meanValue;
        maxValue = _maxValue;
    } else {
        info << "No DEM raster. Stats not updated." << pyre::journal::newline;
    }

    // Announce results
    info << "Min DEM height: " << minValue << pyre::journal::newline
         << "Max DEM height: " << maxValue << pyre::journal::newline
         << "Average DEM height: " << meanValue << pyre::journal::newline;
}

// Compute middle latitude and longitude using reference height
isce3::geometry::DEMInterpolator::cartesian_t
isce3::geometry::DEMInterpolator::
midLonLat() const {
    // Create coordinates for middle X/Y
    cartesian_t xyz{midX(), midY(), _refHeight};

    // Call projection inverse
    return _proj->inverse(xyz);
}

/** @param[in] lon Longitude of interpolation point.
  * @param[in] lat Latitude of interpolation point.
  *
  * Interpolate DEM at a given longitude and latitude */
double isce3::geometry::DEMInterpolator::
interpolateLonLat(double lon, double lat) const {

    // If we don't have a DEM, just return reference height
    double value = _refHeight;
    if (!_haveRaster) {
        return value;
    }

    // Pass latitude and longitude through projection
    cartesian_t xyz;

    const cartesian_t llh{lon, lat, 0.0};
    _proj->forward(llh, xyz);

    // Interpolate DEM at its native coordinates
    value = interpolateXY(xyz[0], xyz[1]);
    // Done
    return value;
}

/** @param[in] x X-coordinate of interpolation point.
  * @param[in] y Y-coordinate of interpolation point.
  *
  * Interpolate DEM at native coordinates */
double isce3::geometry::DEMInterpolator::
interpolateXY(double x, double y) const {

    // If we don't have a DEM, just return reference height
    double value = _refHeight;
    if (!_haveRaster) {
        return value;
    }

    /* Wrap `x` to longitude range [-180, 360] and subsequently
    to DEM coordinates */
    if (_epsgcode == 4326 && (x > 360 || x < -360)) {
        x = std::fmod(x, 360);
    }
    if (_epsgcode == 4326 && x < -180) {
        x += 360;
    }
    if (_epsgcode == 4326 && x - 360 >= _xstart) {
        x -= 360;
    } else if (_epsgcode == 4326 && x < _xstart && x + 360 >= _xstart) {
        x += 360;
    }
    else if (x < _xstart) {
        return _refHeight;
    }

    // Compute the row and column for requested lat and lon
    const double row = (y - _ystart) / _deltay;
    const double col = (x - _xstart) / _deltax;

    // Check validity of interpolation coordinates
    const int irow = int(std::floor(row));
    const int icol = int(std::floor(col));

    // If outside bounds, return reference height
    if (irow < 2 || irow >= int(_dem.length() - 1))
        return _refHeight;
    if (icol < 2 || icol >= int(_dem.width() - 1))
        return _refHeight;

    // Call interpolator and return value
    return _interp->interpolate(col, row, _dem);
}

void isce3::geometry::DEMInterpolator::
validateStatsAccess(const std::string& method) const {
    if (not _haveStats) {
        // Just issue a warning in order to avoid breaking existing code?
        pyre::journal::warning_t warning("isce.core.Geometry");
        warning << "Invalid height stats in use!  Detected call to "
            << "DEMInterpolator::" << method << "() before call to "
            << "DEMInterpolator::computeMinMaxMeanHeight()."
            << pyre::journal::endl;
    }
}
