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
        TopoLayers(size_t width) {
            _lat.resize(width);
            _lon.resize(width);
            _z.resize(width);
            _inc.resize(width);
            _hdg.resize(width);
            _localInc.resize(width);
            _localPsi.resize(width);
            _sim.resize(width);
        }   
        
        // Get array references
        std::valarray<double> & lat() { return _lat; }
        std::valarray<double> & lon() { return _lon; }
        std::valarray<double> & z() { return _z; }
        std::valarray<float> & inc() { return _inc; }
        std::valarray<float> & hdg() { return _hdg; }
        std::valarray<float> & localInc() { return _localInc; }
        std::valarray<float> & localPsi() { return _localPsi; }
        std::valarray<float> & sim() { return _sim; }
        
        // Set values for a single index
        void lat(size_t index, double value) { _lat[index] = value; }
        void lon(size_t index, double value) { _lon[index] = value; }
        void z(size_t index, double value) { _z[index] = value; }
        void inc(size_t index, float value) { _inc[index] = value; }
        void hdg(size_t index, float value) { _hdg[index] = value; }
        void localInc(size_t index, float value) { _localInc[index] = value; }
        void localPsi(size_t index, float value) { _localPsi[index] = value; }
        void sim(size_t index, float value) { _sim[index] = value; }
        
    private:
        // The valarrays for the actual data
        std::valarray<double> _lat;
        std::valarray<double> _lon;
        std::valarray<double> _z;
        std::valarray<float> _inc;
        std::valarray<float> _hdg;
        std::valarray<float> _localInc;
        std::valarray<float> _localPsi;
        std::valarray<float> _sim;
};
    
#endif

// end of file
