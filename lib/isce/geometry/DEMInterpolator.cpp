// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

#include "DEMInterpolator.h"

// Load DEM subset into memory
void isce::geometry::DEMInterpolator::
loadDEM(isce::core::Raster & demRaster,
        double minLon, double maxLon, double minLat, double maxLat,
        isce::core::dataInterpMethod interpMethod) {

    // Initialize journal
    pyre::journal::warning_t warning("isce.core.Geometry");

    // Get original GeoTransform using raster
    double geoTransform[6];
    demRaster.dataset()->GetGeoTransform(geoTransform);
    const double firstlat = geoTransform[3];
    const double firstlon = geoTransform[0];
    const double deltalat = geoTransform[5];
    const double deltalon = geoTransform[1];
    const double lastlat = firstlat + (demRaster.length() - 2) * deltalat;
    const double lastlon = firstlon + (demRaster.width() - 2) * deltalon;

    // Validate requested geographic bounds with input DEM raster
    if (minLon < firstlon) {
        warning << "West limit may be insufficient for global height range"
                << pyre::journal::endl;
        minLon = firstlon;
    }
    if (maxLon > lastlon) {
        warning << "East limit may be insufficient for global height range"
                << pyre::journal::endl;
        maxLon = lastlon;
    }
    if (minLat < lastlat) {
        warning << "South limit may be insufficient for global height range"
                << pyre::journal::endl;
        minLat = lastlat;
    }
    if (maxLat > firstlat) {
        warning << "North limit may be insufficient for global height range"
                << pyre::journal::endl;
        maxLat = firstlat;
    }

    // Compute pixel coordinates for geographic bounds
    int xstart = int((minLon - firstlon) / deltalon);
    int xend = int((maxLon - firstlon) / deltalon + 0.5);
    int ystart = int((maxLat - firstlat) / deltalat);
    int yend = int((minLat - firstlat) / deltalat - 0.5);
    
    // Store actual starting lat/lon for raster subset
    _lonstart = firstlon + xstart * deltalon;
    _latstart = firstlat + ystart * deltalat;
    _deltalat = deltalat;
    _deltalon = deltalon;

    // Resize memory
    const int width = xend - xstart;
    const int length = yend - ystart;
    _dem.resize(length, width);

    // Read data from raster
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < width; ++j) {
            double value;
            demRaster.getValue(value, j+xstart, i+ystart);
            _dem(i,j) = value;
        }
    }
    
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
         << "Top Left: " << _lonstart << " " << _latstart << pyre::journal::newline
         << "Bottom Right: " << _lonstart + _deltalon * (_dem.width() - 1) << " "
         << _latstart + _deltalat * (_dem.length() - 1) << " " << pyre::journal::newline
         << "Spacing: " << _deltalon << " " << _deltalat << pyre::journal::newline
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

// Interpolate DEM
float isce::geometry::DEMInterpolator::
interpolate(double lat, double lon) const {
    // If we don't have a DEM, just return reference height
    float value;
    if (!_haveRaster) {
        value = _refHeight;
    } else {
        // Compute the row and column for requested lat and lon
        const double row = (lat - _latstart) / _deltalat;
        const double col = (lon - _lonstart) / _deltalon;

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
            value = isce::core::Interpolator::akima(_dem.width(), _dem.length(), _dem,
                col, row);
        } else if (_interpMethod == isce::core::BIQUINTIC_METHOD) { 
            value = isce::core::Interpolator::interp_2d_spline(6, _dem, col, row);
        } else if (_interpMethod == isce::core::NEAREST_METHOD) {
            value = _dem(int(std::round(row)), int(std::round(col)));
        }
    }
    return value;
}

// end of file
