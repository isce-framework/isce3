// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-
//

#pragma once

#include "forward.h"

#include <map>

// isce::core
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>

// isce::product
#include <isce3/product/RadarGridParameters.h>

// isce::io
#include <isce3/io/Raster.h>

// isce::geometry
#include <isce3/geometry/geometry.h>

/**
 * Covariance estimation from dual-polarization or quad-polarization data
 */
template<class T>
class isce::signal::Covariance {
public:
    ~Covariance();

    /**
     * Covariance estimation
     *
     * @param[in] slc polarimetric channels provided as std::map of Raster
     * object of polarimetric channels. The keys are two or four of hh, hv,
     * vh, and vv channels.
     * @param[out] cov covariance components obtained by cross multiplication
     * and multi-looking the polarimetric channels
     */
    void covariance(std::map<std::string, isce::io::Raster> & slc,
                    std::map<std::pair<std::string, std::string>,
                             isce::io::Raster> & cov,
                    size_t rangeLooks=1, size_t azimuthLooks=1);

    /**
     * Estimate the Faraday rotation angle from quad-pol data
     *
     * @param[in] slc polarimetric channels
     * @param[out] faradayAngleRaster raster object for Faraday rotation angle
     * @param[in] rangeLooks number of looks in range direction
     * @param[in] azimuthLooks number of looks in azimuth direction
     */
    void faradayRotation(std::map<std::string, isce::io::Raster> & slc,
                         isce::io::Raster & faradayAngleRaster,
                         size_t rangeLooks, size_t azimuthLooks);

    /**
     * Estimate polarimetric orientation angle
     *
     * @param[in] azimuthSlopeRaster raster object of the DEM's slope in azimuth
     * direction
     * @param[in] rangeSlopeRaster raster object of the DEM's slope in range
     * direction
     * @param[in] lookAngleRaster raster object of the look angle
     * @param[out] tauRaster raster object of the polarimetric orientation angle
     */
    void orientationAngle(isce::io::Raster & azimuthSlopeRaster,
                          isce::io::Raster & rangeSlopeRaster,
                          isce::io::Raster & lookAngleRaster,
                          isce::io::Raster & tauRaster);

    /**
     * Geocode covariance components
     *
     * @param[in] rdrCov covariance componenets in radar range-doppler
     * coordinates
     * @param[out] geoCov geocoded covariance componenets
     * @param[in] demRaster digital elevation model (DEM) raster object
     */
    void geocodeCovariance(isce::io::Raster & rdrCov, isce::io::Raster & geoCov,
                           isce::io::Raster & demRaster);

    /**
     * Geocode covariance components. Radiometric Terrain Correction (RTC) is
     * applied before geocoding.
     *
     * @param[in] rdrCov covariance componenets in radar range-doppler
     * coordinates
     * @param[out] geoCov geocoded covariance componenets
     * @param[in] demRaster raster object for digital elevation model (DEM)
     * @param[in] rtcRaster raster object for radiometric terrain correction
     * (RTC) factor
     */
    void geocodeCovariance(isce::io::Raster & rdrCov, isce::io::Raster & geoCov,
                           isce::io::Raster & demRaster,
                           isce::io::Raster & rtc);

    /**
     * Geocode covariance components. RTC and polarimetric orientation angle
     * are applied on Covariance components before geocoding.
     *
     * @param[in] rdrCov covariance componenets in radar range-doppler
     * coordinates
     * @param[out] geoCov geocoded covariance componenets
     * @param[in] demRaster raster object for digital elevation model (DEM)
     * @param[in] rtcRaster raster object for radiometric terrain correction
     * (RTC) factor
     * @param[in] orientationAngleRaster raster object for polarimetric
     * orientation angle
     */
    void geocodeCovariance(isce::io::Raster & rdrCov, isce::io::Raster & geoCov,
                           isce::io::Raster & demRaster, isce::io::Raster & rtc,
                           isce::io::Raster & orientationAngle);

    /**
     * Set geocoded grid.
     *
     * @param[in] geoGridStartX coordinate X of the upper-left corner of the
     * geocoded grid
     * @param[in] geoGridStartY coordinate Y of the upper-left corner of the
     * geocoded grid
     * @param[in] geoGridSpacingX spacing of the geocoded grid in X direction
     * @param[in] geoGridSpacingY spacing of the geocoded grid in Y direction
     * @param[in] geoGridEndX coordinate X of the lower-right corner of the
     * geocoded grid
     * @param[in] geoGridEndY coordinate Y of the lower-right corner of the
     * geocoded grid
     * @param[in] epsgcode EPSG code for defining the projection system
     */
    void geoGrid(double geoGridStartX, double geoGridStartY,
                 double geoGridSpacingX, double geoGridSpacingY,
                 double geoGridEndX, double geoGridEndY, int epsgcode);

    /**
     * Set geocoded grid.
     *
     * @param[in] geoGridStartX coordinate X of the upper-left corner of the
     * geocoded grid
     * @param[in] geoGridStartY coordinate Y of the upper-left corner of the
     * geocoded grid
     * @param[in] geoGridSpacingX spacing of the geocoded grid in X direction
     * @param[in] geoGridSpacingY spacing of the geocoded grid in Y direction
     * @param[in] width number of pixels of the geocoded grid in X direction
     * @param[in] length number of pixels of the geocoded grid in Y direction
     * @param[in] epsgcode EPSG code for defining the projection system
     */
    void geoGrid(double geoGridStartX, double geoGridStartY,
                 double geoGridSpacingX, double geoGridSpacingY, int width,
                 int length, int epsgcode);

    /**
     * Set the input radar grid.
     *
     * @param[in] doppler doppler lookup table
     * @param[in] refEpoch reference epoch
     * @param[in] azimuthStartTime start time of the radar grid in azimuth
     * direction
     * @param[in] azimuthTimeInterval azimuth time interval
     * @param[in] radarGridLength number of pixels of the radar grid in azimuth
     * direction
     * @param[in] startingRange starting slant range of the radar grid
     * @param[in] rangeSpacing range spacing
     * @param[in] side Left or Right
     * @param[in] wavelength Radar wavelength
     * @param[in] radarGridWidth number of pixels of the radar grid in range
     * direction
     */
    void radarGrid(isce::core::LUT2d<double> doppler,
                   isce::core::DateTime refEpoch, double azimuthStartTime,
                   double azimuthTimeInterval, int radarGridLength,
                   double startingRange, double rangeSpacing,
                   isce::core::LookSide side,
                   double wavelength, int radarGridWidth);

    /** Set pulse repetition frequency (PRF). */
    void prf(double p) { _prf = p; }

    /** Set Doppler */
    void doppler(const isce::core::LUT2d<double> & dop) { _doppler = dop; }

    /** Set range sampling frequency */
    void rangeSamplingFrequency(double rngSamplingFreq);

    /** Set range bandwidth*/
    void rangeBandwidth(double rngBandwidth) { _rangeBandwidth = rngBandwidth; }

    /** Set range pixel spacing*/
    void rangePixelSpacing(double rngPixelSpacing) { _rangePixelSpacing = rngPixelSpacing; }

    /** Set radar wavelength*/
    void wavelength(double wvl) { _wavelength = wvl; }

    /** Set interpolator method for geocoding*/
    void interpolator(isce::core::dataInterpMethod method);

    /** Set platform's orbit*/
    void orbit(const isce::core::Orbit & orbit) { _orbit = orbit; }

    /** Set ellipsoid */
    void ellipsoid(const isce::core::Ellipsoid & ellipsoid) { _ellipsoid = ellipsoid; }

    /** Set the projection object*/
    void projection(isce::core::ProjectionBase * proj) { _proj = proj; }

    /** Set the threshold for Geo2rdr computation*/
    void thresholdGeo2rdr(double threshold) { _threshold = threshold; }

    /** Set number of iterations Geo2rdr computation*/
    void numiterGeo2rdr(int numiter) { _numiter = numiter; }

    /** Set lines per block*/
    void linesPerBlock(size_t linesPerBlock) { _linesPerBlock = linesPerBlock; }

    /** Set DEM block margin*/
    void demBlockMargin(double demBlockMargin) { _demBlockMargin = demBlockMargin; }

    /** Set radar block margin*/
    void radarBlockMargin(int radarBlockMargin) { _radarBlockMargin = radarBlockMargin; }

    /** Set interpolator */
    void interpolator(isce::core::Interpolator<T> * interp) { _interp = interp; }

private:
    void _correctRTC(std::valarray<std::complex<float>> & rdrDataBlock,
                     std::valarray<float> & rtcDataBlock);

    void _correctRTC(std::valarray<std::complex<double>> & rdrDataBlock,
                     std::valarray<float> & rtcDataBlock);

    void _computeRangeAzimuthBoundingBox(
            int lineStart, int blockLength, int blockWidth, int margin,
            isce::geometry::DEMInterpolator & demInterp, int & azimuthFirstLine,
            int & azimuthLastLine, int & rangeFirstPixel, int & rangeLastPixel);

    void _loadDEM(isce::io::Raster demRaster,
                  isce::geometry::DEMInterpolator & demInterp,
                  isce::core::ProjectionBase * _proj, int lineStart,
                  int blockLength, int blockWidth, double demMargin);

    void _geo2rdr(double x, double y, double & azimuthTime, double & slantRange,
                  isce::geometry::DEMInterpolator & demInterp);

    void _interpolate(std::valarray<T> & rdrDataBlock,
                      std::valarray<T> & geoDataBlock,
                      std::valarray<double> & radarX,
                      std::valarray<double> & radarY, size_t radarBlockWidth,
                      size_t radarBlockLength, size_t width, size_t length);

    void _faradayRotationAngle(std::valarray<T> & Shh, std::valarray<T> & Shv,
                               std::valarray<T> & Svh, std::valarray<T> & Svv,
                               std::valarray<float> & faradayRotation,
                               size_t width, size_t length, size_t rngLooks,
                               size_t azLooks);

    void _correctFaradayRotation(isce::core::LUT2d<double> & faradayAngle,
                                 std::valarray<std::complex<float>> & Shh,
                                 std::valarray<std::complex<float>> & Shv,
                                 std::valarray<std::complex<float>> & Svh,
                                 std::valarray<std::complex<float>> & Svv,
                                 size_t length, size_t width, size_t lineStart);

    void _orientationAngle(std::valarray<float> & azimuthSlope,
                           std::valarray<float> & rangeSlope,
                           std::valarray<float> & lookAngle,
                           std::valarray<float> & tau);

    void _correctOrientation(std::valarray<float> & tau,
                             std::valarray<std::complex<float>> & C11,
                             std::valarray<std::complex<float>> & C12,
                             std::valarray<std::complex<float>> & C13,
                             std::valarray<std::complex<float>> & C21,
                             std::valarray<std::complex<float>> & C22,
                             std::valarray<std::complex<float>> & C23,
                             std::valarray<std::complex<float>> & C31,
                             std::valarray<std::complex<float>> & C32,
                             std::valarray<std::complex<float>> & C33);

    // pulse repetition frequency
    double _prf;

    // range samping frequency
    double _rangeSamplingFrequency;

    // range signal bandwidth
    double _rangeBandwidth;

    // range pixel spacing
    double _rangePixelSpacing;

    // radar wavelength
    double _wavelength;

    // The following needed for geocoding

    // isce::core objects
    isce::core::Orbit _orbit;
    isce::core::Ellipsoid _ellipsoid;

    // Optimization options
    double _threshold;
    int _numiter;
    size_t _linesPerBlock = 2000;

    // radar grids parameters
    isce::core::LUT2d<double> _doppler;
    isce::product::RadarGridParameters _radarGrid;

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

    // projection object
    isce::core::ProjectionBase * _proj = nullptr;

    // margin around a computed bounding box for DEM (in degrees)
    double _demBlockMargin;

    // margin around the computed bounding box for radar dara (integer number of
    // lines/pixels)
    int _radarBlockMargin;

    // interpolator
    isce::core::Interpolator<T> * _interp = nullptr;

    // RTC correction flag for geocoded covariance
    bool _correctRtcFlag = true;

    // Polarimetric orientation correction flag for geocoded covariance
    bool _correctOrientationFlag = true;

    // dualPol or Quadpol flags
    bool _singlePol = false;
    bool _dualPol = false;
    bool _quadPol = false;

    // coPol and crossPol names (used only for dual-pol)
    std::string _coPol;
    std::string _crossPol;
};

// Get inline implementations for Geocode
#define ISCE_SIGNAL_COVARIANCE_ICC
#include "Covariance.icc"
#undef ISCE_SIGNAL_COVARIANCE_ICC
