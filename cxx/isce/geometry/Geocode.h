//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-

#pragma once

// pyre
#include <pyre/journal.h>

// isce::core
#include <isce/core/Orbit.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/LUT2d.h>
#include <isce/core/Constants.h>

// isce::io
#include <isce/io/Raster.h>

// isce::product
#include <isce/product/Product.h>
#include <isce/product/RadarGridParameters.h>

// isce::geometry
#include "geometry.h"

template<class T>
class isce::geometry::Geocode {
public:

    ~Geocode();

    void geocode(isce::io::Raster & input,
                 isce::io::Raster & output,
                 isce::io::Raster & demRaster);

    /** Set the output geocoded grid*/
    void geoGrid(double geoGridStartX, double geoGridStartY,
                 double geoGridSpacingX, double geoGridSpacingY,
                 double geoGridEndX, double geoGridEndY,
                 int epsgcode);

    void geoGrid(double geoGridStartX, double geoGridStartY,
                 double geoGridSpacingX, double geoGridSpacingY,
                 int width, int length, int epsgcode);

    // Set the input radar grid from Doppler and RadarGridParameters objects
    void radarGrid(isce::core::LUT2d<double> doppler,
                   isce::product::RadarGridParameters rgparam,
                   isce::geometry::Direction lookSide);

    // Set the input radar grid from individual parameters
    void radarGrid(isce::core::LUT2d<double> doppler,
                   isce::core::DateTime refEpoch,
                   double azimuthStartTime,
                   double azimuthTimeInterval,
                   int radarGridLength,
                   double startingRange,
                   double rangeSpacing,
                   double wavelength,
                   int radarGridWidth,
                   isce::geometry::Direction lookSide);

    // Set interpolator
    void interpolator(isce::core::dataInterpMethod method) { _interp = isce::core::createInterpolator<T>(method); }

    void orbit(isce::core::Orbit& orbit) { _orbit = orbit; }

    void ellipsoid(isce::core::Ellipsoid& ellipsoid) { _ellipsoid = ellipsoid; }

    void thresholdGeo2rdr(double threshold) { _threshold = threshold; }

    void numiterGeo2rdr(int numiter) { _numiter = numiter; }

    void linesPerBlock(size_t linesPerBlock) { _linesPerBlock = linesPerBlock; }

    void demBlockMargin(double demBlockMargin) { _demBlockMargin = demBlockMargin; }

    void radarBlockMargin(int radarBlockMargin) { _radarBlockMargin = radarBlockMargin; }

    //interpolator
    void interpolator(isce::core::Interpolator<T> * interp) { _interp = interp; }

private:
    void _loadDEM(isce::io::Raster demRaster,
                  isce::geometry::DEMInterpolator & demInterp,
                  isce::core::ProjectionBase * _proj,
                  int lineStart, int blockLength,
                  int blockWidth, double demMargin);

    void _geo2rdr(double x, double y,
                  double & azimuthTime, double & slantRange,
                  isce::geometry::DEMInterpolator & demInterp,
                  isce::core::ProjectionBase * proj);

    void _interpolate(isce::core::Matrix<T>& rdrDataBlock,
                      isce::core::Matrix<T>& geoDataBlock,
                      std::valarray<double>& radarX,
                      std::valarray<double>& radarY,
                      int rdrBlockWidth, int rdrBlockLength,
                      int azimuthFirstLine, int rangeFirstPixel);

    // isce::core objects
    isce::core::Orbit _orbit;
    isce::core::Ellipsoid _ellipsoid;

    // Optimization options
    double _threshold;
    int _numiter;
    size_t _linesPerBlock = 1000;

    // radar grids parameters
    isce::core::LUT2d<double> _doppler;
    isce::product::RadarGridParameters _radarGrid;
    isce::geometry::Direction _lookSide;

    // start X position for the output geocoded grid
    double _geoGridStartX;

    // start Y position for the output geocoded grid
    double _geoGridStartY;

    // X spacing for the output geocoded grid
    double _geoGridSpacingX;

    // Y spacing for the output geocoded grid
    double _geoGridSpacingY;

    // number of pixels in east-west direction (X direction)
    size_t _geoGridWidth;

    // number of lines in north-south direction (Y direction)
    size_t _geoGridLength;

    // geoTransform array (gdal style)
    double * _geoTrans;

    // epsg code for the output geocoded grid
    int _epsgOut;

    // margin around a computed bounding box for DEM (in degrees)
    double _demBlockMargin;

    // margin around the computed bounding box for radar dara (integer number of lines/pixels)
    int _radarBlockMargin;

    //interpolator
    isce::core::Interpolator<T> * _interp = nullptr;
};

// Get inline implementations for Geocode
#define ISCE_GEOMETRY_GEOCODE_ICC
#include "Geocode.icc"
#undef ISCE_GEOMETRY_GEOCODE_ICC
