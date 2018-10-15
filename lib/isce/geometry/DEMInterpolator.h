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
#include <isce/core/Projections.h>

// isce::io
#include <isce/io/Raster.h>

// Declaration
namespace isce {
    namespace geometry {
        class DEMInterpolator;
    }
}

// DEMInterpolator declaration
class isce::geometry::DEMInterpolator {

    public:
        /** Default constructor with reference height of 0, bilinear interpolation */
        DEMInterpolator() : 
            _haveRaster(false), _refHeight(0.0), _interpMethod(isce::core::BILINEAR_METHOD) {}
        /** Constructor with custom reference height and bilinear interpolation */
        DEMInterpolator(float height) : 
            _haveRaster(false), _refHeight(height), _interpMethod(isce::core::BILINEAR_METHOD) {}
        /** Constructor with custom reference height and custom interpolation method */
        DEMInterpolator(float height, isce::core::dataInterpMethod method) : 
            _haveRaster(false), _refHeight(height), _interpMethod(method) {}

        /** Read in subset of data from a DEM with a supported projection */
        void loadDEM(isce::io::Raster &demRaster,
                     double minLon, double maxLon,
                     double minLat, double maxLat,
                     int epsgcode=4326);

        // Print stats
        void declare() const;

        /** Compute max and mean DEM height */
        void computeHeightStats(float &maxValue, float &meanValue,
                                pyre::journal::info_t &info) const;

        /** Interpolate at a given longitude and latitude */
        double interpolateLonLat(double lon, double lat) const;
        /** Interpolate at native XY coordinates of DEM */
        double interpolateXY(double x, double y) const;

        /** Get starting X coordinate */
        double xStart() const { return _xstart; }
        /** Get starting Y coordinate */
        double yStart() const { return _ystart; }
        /** Get X spacing */
        double deltaX() const { return _deltax; }
        /** Get Y spacing */
        double deltaY() const { return _deltay; }

        /** Get mid X coordinate */
        double midX() const { return _xstart + 0.5*_dem.width()*_deltax; }
        /** Get mid Y coordinate */
        double midY() const { return _ystart + 0.5*_dem.length()*_deltay; }
        /** Get mid longitude and latitude */
        isce::core::cartesian_t midLonLat() const;

        /** Flag indicating whether a DEM raster has been loaded */
        bool haveRaster() const { return _haveRaster; }

        /** Get reference height of interpolator */
        double refHeight() const { return _refHeight; }
        /** Set reference height of interpolator */
        void refHeight(double h) { _refHeight = h; }

        /** Get valarray reference to underlying DEM data */
        std::valarray<float> & data() { return _dem.data(); }

        /** Get width of DEM data used for interpolation */
        inline size_t width() const { return _dem.width(); }
        /** Get length of DEM data used for interpolation */
        inline size_t length() const { return _dem.length(); }

        /** Get EPSG code for input DEM */
        inline int epsgCode() const { return _epsgcode; }

        /** Get interpolator method */
        inline isce::core::dataInterpMethod interpMethod() const {
            return _interpMethod;
        }

    private:
        // Flag indicating whether we have access to a DEM raster
        bool _haveRaster;
        // Constant value if no raster is provided
        float _refHeight;
        // Pointer to a ProjectionBase
        int _epsgcode;
        isce::core::ProjectionBase * _proj;
        // 2D array for storing DEM subset
        isce::core::Matrix<float> _dem;
        // Starting x/y for DEM subset and spacing
        double _xstart, _ystart, _deltax, _deltay;
        // Interpolation method
        isce::core::dataInterpMethod _interpMethod;
};

#endif

// end of file
