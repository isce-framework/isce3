//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-

#ifndef ISCE_GEOMETRY_GEOCODE_H
#define ISCE_GEOMETRY_GEOCODE_H

// pyre
#include <pyre/journal.h>

// isce::core
#include <isce/core/Metadata.h>
#include <isce/core/Orbit.h>
#include <isce/core/Poly2d.h>
#include <isce/core/LUT2d.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Projections.h>
#include <isce/core/Interpolator.h>
#include <isce/core/Constants.h>

// isce::io
#include <isce/io/Raster.h>

// isce::product
#include <isce/product/Product.h>

// isce::geometry
#include "geometry.h"
#include "DEMInterpolator.h"

// Declaration
namespace isce {
    namespace geometry {
        template<class T>
        class Geocode;
    }
}

template<class T>
class isce::geometry::Geocode {

    public:

        Geocode() {};

        ~Geocode() {
 
           if (_interp) {
                delete _interp;
            }
            if (_proj) {
                delete _proj;
            }    
        };

        //inline Geocode(isce::product::Product &);

        void geocode(isce::io::Raster & input, 
                isce::io::Raster & output,
                isce::io::Raster & demRaster);

        /** Set the output geocoded grid*/
        inline void geoGrid(double geoGridStartX, double geoGridStartY,
                double geoGridSpacingX, double geoGridSpacingY,
                double geoGridEndX, double geoGridEndY,
                int epsgcode);
        
        inline void geoGrid(double geoGridStartX, double geoGridStartY, 
                double geoGridSpacingX, double geoGridSpacingY,
                int width, int length, int epsgcode);

        // Set the input radar grid 
        inline void radarGrid(isce::core::LUT2d<double> doppler,
                                double azimuthStartTime,
                                double azimuthTimeInterval,
                                int radarGridLength,
                                double startingRange,
                                double rangeSpacing,
                                double wavelength,
                                int radarGridWidth);

        // Set interpolator 
        inline void interpolator(isce::core::dataInterpMethod method);

        inline void orbit(isce::core::Orbit& orbit);

        inline void orbitInterploationMethod(isce::core::orbitInterpMethod orbitMethod);

        inline void ellipsoid(isce::core::Ellipsoid& ellipsoid);

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

        void _interpolate(isce::core::Matrix<T>& rdrDataBlock, 
                    isce::core::Matrix<T>& geoDataBlock,
                    std::valarray<double>& radarX, std::valarray<double>& radarY,
                    int rdrBlockWidth, int rdrBlockLength);
        

    private:
        
        // isce::core objects
        isce::core::Orbit _orbit;
        isce::core::Ellipsoid _ellipsoid;

        // Optimization options
        double _threshold;
        int _numiter;
        size_t _linesPerBlock = 1000;
        isce::core::orbitInterpMethod _orbitMethod;

        // radar grids parameters
        isce::core::LUT2d<double> _doppler;
        double _azimuthStartTime;
        double _azimuthTimeInterval;
        double _startingRange;
        double _rangeSpacing;
        double _wavelength;
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
#define ISCE_GEOMETRY_GEOCODE_ICC
#include "Geocode.icc"
#undef ISCE_GEOMETRY_GEOCODE_ICC


#endif

