// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018

#pragma once

#include <memory>
#include <string>
#include "forward.h"

// pyre
#include <pyre/journal.h>

// isce3::core
#include <isce3/core/forward.h>
#include <isce3/io/forward.h>

#include <isce3/core/Constants.h>
#include <isce3/core/Interpolator.h>
#include <isce3/error/ErrorCode.h>

// DEMInterpolator declaration
class isce3::geometry::DEMInterpolator {

    using cartesian_t = isce3::core::Vec3;

    public:
        /** Constructor with custom reference height and bilinear interpolation */
        inline DEMInterpolator(float height = 0.0, int epsg = 4326) :
            _haveRaster{false},
            _refHeight{height},
            _haveStats{true},
            _minValue{height},
            _meanValue{height},
            _maxValue{height},
            _epsgcode{epsg},
            _interpMethod{isce3::core::BILINEAR_METHOD} {}

        /** Constructor with custom reference height and custom interpolator method */
        inline DEMInterpolator(float height,
                               isce3::core::dataInterpMethod method,
                               int epsg = 4326) :
            _haveRaster{false},
            _refHeight{height},
            _haveStats{true},
            _minValue{height},
            _meanValue{height},
            _maxValue{height},
            _epsgcode{epsg},
            _interpMethod{method} {}

        /** Read in subset of data from a DEM with a supported projection
        * @param[in]  dem_raster              DEM raster
        * @param[in]  minX                    Minimum X/easting position
        * @param[in]  maxX                    Maximum X/easting position
        * @param[in]  minY                    Minimum Y/northing position
        * @param[in]  maxY                    Maximum Y/northing position
        * @param[in]  dem_raster_band         DEM raster band (starting from 1)
        */
        isce3::error::ErrorCode loadDEM(isce3::io::Raster& demRaster,
                double min_x, double max_x, double min_y, double max_y,
                const int dem_raster_band = 1);

        /** Read in entire DEM with a supported projection
        * @param[in]  dem_raster              DEM raster
        * @param[in]  dem_raster_band         DEM raster band (starting from 1)
        */
        void loadDEM(isce3::io::Raster &demRaster,
                     const int dem_raster_band = 1);

        // Print stats
        void declare() const;

        /** Compute min, max, and mean DEM height
         * @param[out] minValue Minimum DEM height
         * @param[out] maxValue Maximum DEM height
         * @param[out] meanValue Mean DEM height
         * 
         * If stats have already been computed then no calculation is done.
         */
        void computeMinMaxMeanHeight(float &minValue, float &maxValue,
                                     float &meanValue);

        /** Interpolate at a given longitude and latitude */
        double interpolateLonLat(double lon, double lat) const;
        /** Interpolate at native XY coordinates of DEM */
        double interpolateXY(double x, double y) const;

        /** Get starting X coordinate */
        double xStart() const { return _xstart; }
        /** Set starting X coordinate */
        void xStart(double xstart) { _xstart = xstart; }

        /** Get starting Y coordinate */
        double yStart() const { return _ystart; }
        /** Set starting Y coordinate */
        void yStart(double ystart) { _ystart = ystart; }

        /** Get X spacing */
        double deltaX() const { return _deltax; }
        /** Set X spacing */
        void deltaX(double deltax) { _deltax = deltax; }

        /** Get Y spacing */
        double deltaY() const { return _deltay; }
        /** Set Y spacing */
        void deltaY(double deltay) { _deltay = deltay; }

        /** Get mid X coordinate */
        double midX() const { return _xstart + 0.5*width()*_deltax; }
        /** Get mid Y coordinate */
        double midY() const { return _ystart + 0.5*length()*_deltay; }
        /** Get mid longitude and latitude */
        cartesian_t midLonLat() const;

        /** Flag indicating whether a DEM raster has been loaded */
        bool haveRaster() const { return _haveRaster; }

        /** Get reference height of interpolator */
        double refHeight() const { return _refHeight; }
        /** Set reference height of interpolator */
        void refHeight(double h) {
            _refHeight = h;
            if (not haveRaster()) {
                _minValue = h;
                _meanValue = h;
                _maxValue = h;
            }
        }

        /** Flag indicating if stats are already known. */
        bool haveStats() const { return _haveStats; }

        /** Get mean height value */
        inline float meanHeight() const {
            validateStatsAccess("meanHeight");
            return _meanValue;
        }

        /** Get max height value */
        inline float maxHeight() const {
            validateStatsAccess("maxHeight");
            return _maxValue;
        }

        /** Get min height value */
        inline float minHeight() const {
            validateStatsAccess("minHeight");
            return _minValue;
        }

        /** Get pointer to underlying DEM data */
        float * data() { return _dem.data(); }

        /** Get pointer to underlying DEM data */
        const float* data() const { return _dem.data(); }

        /** Get width of DEM data used for interpolation */
        inline size_t width() const { return (_haveRaster ? _dem.width() : _width); }
        /** Set width of DEM data used for interpolation */
        inline void width(int width) { _width = width; }

        /** Get length of DEM data used for interpolation */
        inline size_t length() const { return (_haveRaster ? _dem.length() : _length); }
        /** Set length of DEM data used for interpolation */
        inline void length(int length) { _length = length; }

        /** Get EPSG code for input DEM */
        inline int epsgCode() const { return _epsgcode; }
        /** Set EPSG code for input DEM */
        void epsgCode(int epsgcode);

        /** Get Pointer to a ProjectionBase */
        inline isce3::core::ProjectionBase* proj() const {return _proj.get(); }

        /** Get interpolator method enum */
        inline isce3::core::dataInterpMethod interpMethod() const {
            return _interpMethod;
        }
        /** Set interpolator method enum */
        inline void interpMethod(isce3::core::dataInterpMethod interpMethod) {
            _interpMethod = interpMethod;
        }

    private:
        // Flag indicating whether we have access to a DEM raster
        bool _haveRaster;
        // Constant value if no raster is provided
        float _refHeight;
        // Statistics
        bool _haveStats;
        float _minValue;
        float _meanValue;
        float _maxValue;
        // Pointer to a ProjectionBase
        int _epsgcode;
        std::shared_ptr<isce3::core::ProjectionBase> _proj;
        // Pointer to an Interpolator
        isce3::core::dataInterpMethod _interpMethod;
        std::shared_ptr<isce3::core::Interpolator<float>> _interp;
        // 2D array for storing DEM subset
        isce3::core::Matrix<float> _dem;
        // Starting x/y for DEM subset and spacing
        double _xstart, _ystart, _deltax, _deltay;
        int _width, _length;

        // Check if stats are accessed before they're computed.
        void validateStatsAccess(const std::string& method) const;
};
