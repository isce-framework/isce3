// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_DEMInterpolator_H
#define ISCE_CORE_DEMInterpolator_H

// pyre
#include <portinfo>
#include <pyre/journal.h>

// isce::core
#include <isce/core/Constants.h>
#include <isce/core/Interpolator.h>
#include <isce/core/Raster.h>

// Declaration
namespace isce {
    namespace geometry {
        class DEMInterpolator;
    }
}

// DEMInterpolator declaration
class isce::geometry::DEMInterpolator {

    public:
        // Constructors
        DEMInterpolator() : _haveRaster(false), _refHeight(0.0) {}
        DEMInterpolator(float height) : _haveRaster(false), _refHeight(height) {}

        // Read in subset of data
        void loadDEM(isce::core::Raster &, double, double, double, double,
                     isce::core::dataInterpMethod);
        // Print stats
        void declare() const;
        // Compute max and mean DEM height
        void computeHeightStats(float &, float &, pyre::journal::info_t &) const;
        // Interpolate at a given latitude and longitude
        double interpolate(double, double) const;
        // Get transform properties
        double lonStart() const { return _lonstart; }
        double latStart() const { return _latstart; }
        double deltaLon() const { return _deltalon; }
        double deltaLat() const { return _deltalat; }
        // Middle latitude and longitude
        double midLon() const { return _lonstart + 0.5*_dem.width()*_deltalon; }
        double midLat() const { return _latstart + 0.5*_dem.length()*_deltalat; }

    private:
        // Flag indicating whether we have access to a DEM raster
        bool _haveRaster;
        // Constant value if no raster is provided
        float _refHeight;
        // 2D array for storing DEM subset
        isce::core::Matrix<float> _dem;
        // Starting lat/lon for DEM subset and spacing
        double _lonstart, _latstart, _deltalon, _deltalat;
        // Interpolation method
        isce::core::dataInterpMethod _interpMethod;
};

#endif

// end of file
