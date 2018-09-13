// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

#include "DEMInterpolator.h"

// Load DEM subset into memory
void isce::geometry::DEMInterpolator::
loadDEM(isce::io::Raster & demRaster,
        double minLon, double maxLon, double minLat, double maxLat,
        isce::core::dataInterpMethod interpMethod, int epsgcode) {

    // Initialize journal
    pyre::journal::warning_t warning("isce.core.Geometry");

    // Get original GeoTransform using raster
    double geoTransform[6];
    demRaster.getGeoTransform(geoTransform);
    const double deltaY = geoTransform[5];
    const double deltaX = geoTransform[1];
    // Use center of pixel as starting coordinate
    const double firstY = geoTransform[3] + 0.5 * deltaY;
    const double firstX = geoTransform[0] + 0.5 * deltaX;
    // Compute ending coordinate 
    const double lastY = firstY + (demRaster.length() - 2) * deltaY;
    const double lastX = firstX + (demRaster.width() - 2) * deltaX;

    // Check EPSG code; if -9999, assume 4326
    if (epsgcode == -9999) {
        epsgcode = 4326;
    }
    // Initialize projection
    _epsgcode = epsgcode;
    _proj = isce::core::createProj(epsgcode);

    // Convert min longitude and latitude to XY coordinates of DEM
    isce::core::cartesian_t llh{minLon, minLat, 0.0};
    isce::core::cartesian_t xyz;
    _proj->forward(llh, xyz);
    double minX = xyz[0];
    double minY = xyz[1];

    // Convert max longitude and latitude to XY coordinates of DEM
    llh = {maxLon, maxLat, 0.0};
    _proj->forward(llh, xyz);
    double maxX = xyz[0];
    double maxY = xyz[1];

    // Validate requested geographic bounds with input DEM raster
    if (minX < firstX) {
        warning << "West limit may be insufficient for global height range"
                << pyre::journal::endl;
        minX = firstX;
    }
    if (maxX > lastX) {
        warning << "East limit may be insufficient for global height range"
                << pyre::journal::endl;
        maxX = lastX;
    }
    if (minY < lastY) {
        warning << "South limit may be insufficient for global height range"
                << pyre::journal::endl;
        minY = lastY;
    }
    if (maxY > firstY) {
        warning << "North limit may be insufficient for global height range"
                << pyre::journal::endl;
        maxY = firstY;
    }

    // Compute pixel coordinates for geographic bounds
    int xstart = int((minX - firstX) / deltaX);
    int xend = int((maxX - firstX) / deltaX + 0.5);
    int ystart = int((maxY - firstY) / deltaY);
    int yend = int((minY - firstY) / deltaY - 0.5);
    
    // Store actual starting lat/lon for raster subset
    _xstart = firstX + xstart * deltaX;
    _ystart = firstY + ystart * deltaY;
    _deltax = deltaX;
    _deltay = deltaY;

    // Resize memory
    const int width = xend - xstart;
    const int length = yend - ystart;
    _dem.resize(length, width);

    // Read in the DEM
    demRaster.getBlock(_dem.data(), xstart, ystart, width, length);

    // Indicate we have loaded a valid raster
    _haveRaster = true;
    // Store interpolation method
    _interpMethod = interpMethod;
}

// Load DEM subset into memory
void isce::geometry::DEMInterpolator::
declare() const {
    pyre::journal::info_t info("isce.core.DEMInterpolator");
    info << pyre::journal::newline
         << "Actual DEM bounds used:" << pyre::journal::newline
         << "Top Left: " << _xstart << " " << _ystart << pyre::journal::newline
         << "Bottom Right: " << _xstart + _deltax * (_dem.width() - 1) << " "
         << _ystart + _deltay * (_dem.length() - 1) << " " << pyre::journal::newline
         << "Spacing: " << _deltax << " " << _deltay << pyre::journal::newline
         << "Dimensions: " << _dem.width() << " " << _dem.length() << pyre::journal::newline;
}

// Compute maximum DEM height
void isce::geometry::DEMInterpolator::
computeHeightStats(float & maxValue, float & meanValue, pyre::journal::info_t & info) const {
    // Announce myself
    info << "Computing DEM statistics" << pyre::journal::newline << pyre::journal::newline;
    // If we don't have a DEM, just use reference height
    if (!_haveRaster) {
        maxValue = _refHeight;
        meanValue = _refHeight;
    } else {
        maxValue = -10000.0;
        float sum = 0.0;
        for (int i = 0; i < int(_dem.length()); ++i) {
            for (int j = 0; j < int(_dem.width()); ++j) {
                float value = _dem(i,j);
                if (value > maxValue)
                    maxValue = value;
                sum += value;
            }
        }
        meanValue = sum / (_dem.width() * _dem.length());
    }
    // Announce results
    info << "Max DEM height: " << maxValue << pyre::journal::newline
         << "Average DEM height: " << meanValue << pyre::journal::newline;
}

// Compute middle latitude and longitude using reference height
isce::core::cartesian_t
isce::geometry::DEMInterpolator::
midLonLat() const {

    // Create coordinates for middle X/Y
    isce::core::cartesian_t xyz{midX(), midY(), _refHeight};

    // Call projection inverse
    isce::core::cartesian_t llh;
    _proj->inverse(xyz, llh);

    // Done
    return llh;
}

// Interpolate DEM at a given longitude and latitude (in radians)
double isce::geometry::DEMInterpolator::
interpolateLonLat(double lon, double lat) const {

    // If we don't have a DEM, just return reference height
    double value = _refHeight;
    if (!_haveRaster) {
        return value;
    }

    // Pass latitude and longitude through projection
    isce::core::cartesian_t xyz;
    const isce::core::cartesian_t llh{lon, lat, 0.0};
    _proj->forward(llh, xyz);

    // Interpolate DEM at its native coordinates
    value = interpolateXY(xyz[0], xyz[1]);
    // Done
    return value;
}

// Interpolate DEM at native coordinates
double isce::geometry::DEMInterpolator::
interpolateXY(double x, double y) const {

    // If we don't have a DEM, just return reference height
    double value = _refHeight;
    if (!_haveRaster) {
        return value;
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
    
    // Choose correct interpolation routine
    if (_interpMethod == isce::core::BILINEAR_METHOD) {
        value = isce::core::Interpolator::bilinear(col, row, _dem);
    } else if (_interpMethod == isce::core::BICUBIC_METHOD) {
        value = isce::core::Interpolator::bicubic(col, row, _dem);
    } else if (_interpMethod == isce::core::AKIMA_METHOD) {
        value = isce::core::Interpolator::akima(col, row, _dem);
    } else if (_interpMethod == isce::core::BIQUINTIC_METHOD) { 
        value = isce::core::Interpolator::interp_2d_spline(6, _dem, col, row);
    } else if (_interpMethod == isce::core::NEAREST_METHOD) {
        value = _dem(int(std::round(row)), int(std::round(col)));
    }
    // Done
    return value;
}

// end of file
