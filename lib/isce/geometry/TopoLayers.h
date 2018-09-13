//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_GEOMETRY_TOPOLAYERS_H
#define ISCE_GEOMETRY_TOPOLAYERS_H

#include <valarray>

// Declaration
namespace isce {
    namespace geometry {
        class TopoLayers;
    }
}

class isce::geometry::TopoLayers {

    public:
        // Constructors
        TopoLayers(size_t length, size_t width) : _length(length), _width(width) {
            _x.resize(length*width);
            _y.resize(length*width);
            _z.resize(length*width);
            _inc.resize(length*width);
            _hdg.resize(length*width);
            _localInc.resize(length*width);
            _localPsi.resize(length*width);
            _sim.resize(length*width);
        }   

        // Get sizes
        inline size_t length() const { return _length; }
        inline size_t width() const { return _width; }

        // Get array references
        std::valarray<double> & x() { return _x; }
        std::valarray<double> & y() { return _y; }
        std::valarray<double> & z() { return _z; }
        std::valarray<float> & inc() { return _inc; }
        std::valarray<float> & hdg() { return _hdg; }
        std::valarray<float> & localInc() { return _localInc; }
        std::valarray<float> & localPsi() { return _localPsi; }
        std::valarray<float> & sim() { return _sim; }
        
        // Set values for a single index
        void x(size_t row, size_t col, double value) {
            _x[row*_width+col] = value;
        }
        
        void y(size_t row, size_t col, double value) {
            _y[row*_width + col] = value;
        }
        
        void z(size_t row, size_t col, double value) {
            _z[row*_width + col] = value;
        }
        
        void inc(size_t row, size_t col, float value) {
            _inc[row*_width + col] = value;
        }
        
        void hdg(size_t row, size_t col, float value) {
            _hdg[row*_width + col] = value;
        }
        
        void localInc(size_t row, size_t col, float value) {
            _localInc[row*_width + col] = value;
        }
        
        void localPsi(size_t row, size_t col, float value) {
            _localPsi[row*_width + col] = value;
        }
    
        void sim(size_t row, size_t col, float value) {
            _sim[row*_width + col] = value;
        }
        
    private:
        // The valarrays for the actual data
        std::valarray<double> _x;
        std::valarray<double> _y;
        std::valarray<double> _z;
        std::valarray<float> _inc;
        std::valarray<float> _hdg;
        std::valarray<float> _localInc;
        std::valarray<float> _localPsi;
        std::valarray<float> _sim;

        // Dimensions
        size_t _length, _width;
};
    
#endif

// end of file
