// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#ifndef ISCE_LIB_COVARIANCE_H
#define ISCE_LIB_COVARIANCE_H

#include <map>

// isce::core
#include <isce/core/Metadata.h>
#include <isce/core/Orbit.h>
#include <isce/core/Poly2d.h>
#include <isce/core/LUT1d.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Projections.h>
#include <isce/core/Interpolator.h>

// isce::product
#include <isce/product/Product.h>

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

template<class T>
class isce::signal::Covariance {

    public:

        Covariance() {};

        ~Covariance() {
        
            if (_interp) {
                delete _interp;
            }
            if (_proj) {
                delete _proj;
            }
        };

        // covariance estimation 
        void covariance(std::map<std::string, isce::io::Raster> & slc,
                std::map<std::pair<std::string, std::string>, isce::io::Raster> & cov);

        void geocodeCovariance(
                    isce::io::Raster& rdrCov,
                    isce::io::Raster& rtc,
                    isce::io::Raster & demRaster,
                    isce::io::Raster& geoCov);

        inline void geoGrid(double geoGridStartX, double geoGridStartY,
                double geoGridSpacingX, double geoGridSpacingY,
                double geoGridEndX, double geoGridEndY,
                int epsgcode);

	inline void geoGrid(double geoGridStartX, double geoGridStartY, 
                double geoGridSpacingX, double geoGridSpacingY,
                int width, int length, int epsgcode);
        
        // Set the input radar grid 
        inline void radarGrid(isce::core::LUT1d<double> doppler,
                                isce::core::DateTime azimuthStartTime,
                                double azimuthTimeInterval,
                                int radarGridLength,
                                double startingRange,
                                double rangeSpacing,
                                int radarGridWidth);

        // Set interpolator 

        inline void numberOfRangeLooks(int rngLooks);

        inline void numberOfAzimuthLooks(int azimuthLooks);

        inline void prf(double p);

        inline void doppler(isce::core::LUT1d<double> dop);

        inline void rangeSamplingFrequency(double rngSamplingFreq);

        inline void rangeBandwidth(double rngBandwidth);

        inline void rangePixelSpacing(double rngPixelSpacing);

        inline void wavelength(double wvl);
        
        inline void interpolator(isce::core::dataInterpMethod method);

        inline void orbit(isce::core::Orbit& orbit, isce::core::DateTime refEpoch);

        inline void orbitInterploationMethod(isce::core::orbitInterpMethod orbitMethod);

        inline void ellipsoid(isce::core::Ellipsoid& ellipsoid);

        inline void mode(isce::product::ImageMode& mode);

        inline void projection(isce::core::ProjectionBase * proj);

        inline void thresholdGeo2rdr(double threshold);

        inline void numiterGeo2rdr(int numiter);

        inline void linesPerBlock(size_t linesPerBlock);

        inline void demBlockMargin(double demBlockMargin);

        inline void radarBlockMargin(int radarBlockMargin);

        //interpolator
        //isce::core::Interpolator * _interp = nullptr;
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

        /*
        void _interpolate(isce::core::Matrix<T>& rdrDataBlock, 
                    isce::core::Matrix<T>& geoDataBlock,
                    std::valarray<double>& radarX, std::valarray<double>& radarY,
                    int rdrBlockWidth, int rdrBlockLength);
        */

        void _interpolate(std::valarray<T>& rdrDataBlock,
                    std::valarray<T>& geoDataBlock,
                    std::valarray<double>& radarX, std::valarray<double>& radarY,
                    size_t radarBlockWidth, size_t radarBlockLength,
                    size_t width, size_t length);

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

        // isce::product objects
        isce::product::ImageMode _mode;
    
        // Optimization options
        double _threshold;
        int _numiter;
        size_t _linesPerBlock = 1000;
        isce::core::orbitInterpMethod _orbitMethod;

        // radar grids parameters
        isce::core::LUT1d<double> _doppler;
        isce::core::DateTime _azimuthStartTime, _refEpoch;
        double _azimuthTimeInterval;
        double _startingRange;
        double _rangeSpacing;
        int _radarGridLength;
        int _radarGridWidth;

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
        isce::core::ProjectionBase * _proj;        

        // margin around a computed bounding box for DEM (in degrees)
        double _demBlockMargin;

        // margin around the computed bounding box for radar dara (integer number of lines/pixels)
        int _radarBlockMargin;

        //interpolator 
        isce::core::Interpolator<T> * _interp = nullptr;

};

// Get inline implementations for Geocode
#define ISCE_SIGNAL_COVARIANCE_ICC
#include "Covariance.icc"
#undef ISCE_SIGNAL_COVARIANCE_ICC


#endif

