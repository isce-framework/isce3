//-*- C++ -*-
//-*- coding: utf-8 -*-

#pragma once

namespace isce {
    namespace product {
        class GeoGridParameters; 
    }
}

class isce::product::GeoGridParameters {

    public:

        //inline GeoGridParameters() {}

        inline GeoGridParameters(double geoGridStartX, double geoGridStartY,
                double geoGridSpacingX, double geoGridSpacingY,
                int width, int height, int epsgcode);

        inline void startX(double & x0) {_startX = x0;}

        inline void startY(double & y0) {_startY = y0;}

        inline void spacingX(double & dx) {_spacingX = dx;}

        inline void spacingY(double & dy) {_spacingY = dy;}

        inline void length(int & l) {_length = l;};

        inline void width(int & w) {_width = w;};

        inline void epsg(int & e) {_epsg = e;};

        /** Get */
        inline double startX() const { return _startX; };

        /** Get */
        inline double startY() const { return _startY; };

        /** Get */
        inline double spacingX() const { return _spacingX; };

        /** Get */
        inline double spacingY() const { return _spacingY; };

        inline int width() const { return _width; };

        inline int length() const { return _length; };

        inline int epsg() const {return _epsg; };

        inline double* geotransform() const {return _geoTrans; }; 

    protected:
        // start X position for the geocoded grid
        double _startX = 0.0;

        // start Y position for the geocoded grid
        double _startY = 0.0;

        // X spacing for the geocoded grid
        double _spacingX = 0.0;

        // Y spacing for the geocoded grid
        double _spacingY = 0.0;

        // number of pixels in east-west direction (X direction)
        int _width = 0.0;

        // number of lines in north-south direction (Y direction)
        int _length = 0.0;

        // geoTransform array (gdal style)
        double * _geoTrans=new double[6];

        // epsg code for the output geocoded grid
        int _epsg;

};

isce::product::GeoGridParameters::
GeoGridParameters(double geoGridStartX, double geoGridStartY,
                double geoGridSpacingX, double geoGridSpacingY,
                int width, int length, int epsgcode) :
    // Assumption: origin is the top-left corner of the top-left pixel of the grid
    // the starting coordinate of the output geocoded grid in X direction.
    // Since the input is alwayas referring to the top-left corner of the
    // top-left pixel, we adjust to the center for internal use only
    _startX(geoGridStartX + geoGridSpacingX/2),

    // the starting coordinate of the output geocoded grid in Y direction.
    // adjusted to the center of the pixel for internal use only
    _startY(geoGridStartY + geoGridSpacingY/2),

    // spacing of the output geocoded grid in X
    _spacingX(geoGridSpacingX),

    // spacing of the output geocoded grid in Y
    _spacingY(geoGridSpacingY),

    // number of lines (rows) in the geocoded grid (Y direction)
    _width(width),

    // number of columns in the geocoded grid (Y direction)
    _length(length),

    // Save the EPSG code
    _epsg(epsgcode) {
        // Assumption: origin is the top-left corner of the top-left pixel of the grid
        _geoTrans[0] = geoGridStartX;
        _geoTrans[1] = geoGridSpacingX;
        _geoTrans[2] = 0.0;
        _geoTrans[3] = geoGridStartY;
        _geoTrans[4] = 0.0;
        _geoTrans[5] = geoGridSpacingY;}
     

