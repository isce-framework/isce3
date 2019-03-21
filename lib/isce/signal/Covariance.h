// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-
//

#ifndef ISCE_LIB_COVARIANCE_H
#define ISCE_LIB_COVARIANCE_H

#include <map>

// isce::core
#include <isce/core/Metadata.h>
#include <isce/core/Orbit.h>
#include <isce/core/Poly2d.h>
#include <isce/core/LUT2d.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Projections.h>
#include <isce/core/Interpolator.h>
#include <isce/core/LUT2d.h>

// isce::product
#include <isce/product/Product.h>
#include <isce/product/RadarGridParameters.h>

// isce::io
#include <isce/io/Raster.h>

// isce::geometry
#include <isce/geometry/geometry.h>

#include <isce/geometry/DEMInterpolator.h>

#include "Signal.h"
#include "Looks.h"
#include "Crossmul.h"

namespace isce {
    namespace signal {
        template<class T>
        class Covariance;
    }
}

/** \brief Covariance estimation from dual-polarization or quad-polarization data 
 *
 */
template<class T>
class isce::signal::Covariance {

    public:
        
        // constructor
        Covariance() {};

        // destructor
        ~Covariance() {
        
            if (_interp) {
                delete _interp;
            }
            if (_proj) {
                delete _proj;
            }
        };
        

        /** Covariance estimation */
        void covariance(std::map<std::string, isce::io::Raster> & slc,
                    std::map<std::pair<std::string, std::string>, isce::io::Raster> & cov);

        /** Estimate the Faraday rotation angle from quad-pol data*/
        void faradayRotation(std::map<std::string, isce::io::Raster> & slc,
                    isce::io::Raster & faradayAngleRaster,
                    size_t rangeLooks, size_t azimuthLooks);

        
        /** Correct Faraday rotation for quad-pol data */
        /*
        void correctFaradayRotation(std::map<std::string, isce::io::Raster> & slc,
                    isce::core::LUT2d & faradayAngle,
                    std::map<std::string, isce::io::Raster> & correctedSlc);
        */

        /** Estimate polarimetric orientation angle */
        void orientationAngle(isce::io::Raster& azimuthSlopeRaster,
                    isce::io::Raster& rangeSlopeRaster,
                    isce::io::Raster& lookAngleRaster,
                    isce::io::Raster& tau);

        /** Geocode covariance components*/
        void geocodeCovariance(
                    isce::io::Raster& rdrCov,
                    isce::io::Raster& geoCov,
                    isce::io::Raster& demRaster);

        /** Geocode covariance components. Radiometric Terrain Correction (RTC) is applied before geocoding. */
        void geocodeCovariance(
                    isce::io::Raster& rdrCov,
                    isce::io::Raster& geoCov,
                    isce::io::Raster& demRaster,
                    isce::io::Raster& rtc);

        /** Geocode covariance components. RTC and polarimetric orientation angle are applied on Covariance components before geocoding. */
        void geocodeCovariance(
                    isce::io::Raster& rdrCov,
                    isce::io::Raster& geoCov,
                    isce::io::Raster& demRaster,
                    isce::io::Raster& rtc,
                    isce::io::Raster& orientationAngle);

        /** Set geocoded grid. */
        inline void geoGrid(double geoGridStartX, double geoGridStartY,
                double geoGridSpacingX, double geoGridSpacingY,
                double geoGridEndX, double geoGridEndY,
                int epsgcode);

        /** Set geocoded grid. */
	inline void geoGrid(double geoGridStartX, double geoGridStartY, 
                double geoGridSpacingX, double geoGridSpacingY,
                int width, int length, int epsgcode);
        
        /** Set the input radar grid. */
        inline void radarGrid(isce::core::LUT2d<double> doppler,
                    isce::core::DateTime refEpoch,
                    double azimuthStartTime,
                    double azimuthTimeInterval,
                    int radarGridLength,
                    double startingRange,
                    double rangeSpacing,
                    double wavelength,
                    int radarGridWidth);

        /** Set number of looks in range direction for covariance estimation */
        inline void numberOfRangeLooks(int rngLooks);

        /** Set number of looks in azimuth direction for covariance estimation */
        inline void numberOfAzimuthLooks(int azimuthLooks);

        /** Set pulse repetition frequency (PRF). */
        inline void prf(double p);

        /** Set Doppler */
        inline void doppler(isce::core::LUT2d<double> dop);

        /** Set range sampling frequency */
        inline void rangeSamplingFrequency(double rngSamplingFreq);

        /** Set range bandwidth*/
        inline void rangeBandwidth(double rngBandwidth);
        
        /** Set range pixel spacing*/
        inline void rangePixelSpacing(double rngPixelSpacing);

        /** Set radar wavelength*/
        inline void wavelength(double wvl);
        
        /** Set interpolator method for geocoding*/
        inline void interpolator(isce::core::dataInterpMethod method);

        /** Set platform's orbit*/
        inline void orbit(isce::core::Orbit& orbit);

        /** Set orbit interploation method*/
        inline void orbitInterploationMethod(isce::core::orbitInterpMethod orbitMethod);

        /** Set ellipsoid */
        inline void ellipsoid(isce::core::Ellipsoid& ellipsoid);

        /** Set the projection object*/
        inline void projection(isce::core::ProjectionBase * proj);

        /** Set the threshold for Geo2rdr computation*/
        inline void thresholdGeo2rdr(double threshold);

        /** Set number of iterations Geo2rdr computation*/
        inline void numiterGeo2rdr(int numiter);

        /** Set lines per block*/
        inline void linesPerBlock(size_t linesPerBlock);

        /** Set DEM block margin*/
        inline void demBlockMargin(double demBlockMargin);

        /** Set radar block margin*/
        inline void radarBlockMargin(int radarBlockMargin);

        /** Set interpolator */
        inline void interpolator(isce::core::Interpolator<T> * interp);

    private:

        void _correctRTC(std::valarray<std::complex<float>> & rdrDataBlock,
                    std::valarray<float> & rtcDataBlock);

        void _correctRTC(std::valarray<std::complex<double>> & rdrDataBlock,
                    std::valarray<float> & rtcDataBlock);


        void _computeRangeAzimuthBoundingBox(int lineStart, 
                    int blockLength, int blockWidth,
                    int margin, isce::geometry::DEMInterpolator & demInterp,
                    int & azimuthFirstLine, int & azimuthLastLine,
                    int & rangeFirstPixel, int & rangeLastPixel);

        void _loadDEM(isce::io::Raster demRaster,
                    isce::geometry::DEMInterpolator & demInterp,
                    isce::core::ProjectionBase * _proj,
                    int lineStart, int blockLength,
                    int blockWidth, double demMargin);

        void _geo2rdr(double x, double y,
                    double & azimuthTime, double & slantRange,
                    isce::geometry::DEMInterpolator & demInterp);

        void _interpolate(std::valarray<T>& rdrDataBlock,
                    std::valarray<T>& geoDataBlock,
                    std::valarray<double>& radarX, std::valarray<double>& radarY,
                    size_t radarBlockWidth, size_t radarBlockLength,
                    size_t width, size_t length);

        void _faradayRotationAngle(std::valarray<T>& Shh,
                    std::valarray<T>& Shv,
                    std::valarray<T>& Svh,
                    std::valarray<T>& Svv,
                    std::valarray<float>& faradayRotation,
                    size_t width, size_t length,
                    size_t rngLooks, size_t azLooks);

        void _correctFaradayRotation(isce::core::LUT2d<double>& faradayAngle,
                    std::valarray<std::complex<float>>& Shh,
                    std::valarray<std::complex<float>>& Shv,
                    std::valarray<std::complex<float>>& Svh,
                    std::valarray<std::complex<float>>& Svv,
                    size_t length,
                    size_t width,
                    size_t lineStart);

        void _orientationAngle(std::valarray<float>& azimuthSlope,
                    std::valarray<float>& rangeSlope,
                    std::valarray<float>& lookAngle,
                    std::valarray<float>& tau);

        void _correctOrientation(std::valarray<float>& tau,
                    std::valarray<std::complex<float>>& C11,
                    std::valarray<std::complex<float>>& C12,
                    std::valarray<std::complex<float>>& C13,
                    std::valarray<std::complex<float>>& C21,
                    std::valarray<std::complex<float>>& C22,
                    std::valarray<std::complex<float>>& C23,
                    std::valarray<std::complex<float>>& C31,
                    std::valarray<std::complex<float>>& C32,
                    std::valarray<std::complex<float>>& C33);

    private:

        // following members are needed for crossmul
         
        // number of range looks
        int _rangeLooks = 1;

        // number of azimuth looks
        int _azimuthLooks = 1;

        //pulse repetition frequency
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
        isce::core::orbitInterpMethod _orbitMethod;

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

        // margin around the computed bounding box for radar dara (integer number of lines/pixels)
        int _radarBlockMargin;

        // interpolator 
        isce::core::Interpolator<T> * _interp = nullptr;

        // RTC correction flag for geocoded covariance
        bool _correctRtcFlag = true;

        // Polarimetric orientation correction flag for geocoded covariance
        bool _correctOrientationFlag = true;

        // dualPol or Quadpol flags
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


#endif

