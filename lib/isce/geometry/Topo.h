//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan Riel
// Copyright 2017-2018

#ifndef ISCE_CORE_TOPO_H
#define ISCE_CORE_TOPO_H

// pyre
#include <portinfo>
#include <pyre/journal.h>

// isce::core
#include "isce/core/Metadata.h"
#include "isce/core/Orbit.h"
#include "isce/core/Poly2d.h"
#include "isce/core/Ellipsoid.h"
#include "isce/core/Raster.h"

// isce::geometry
#include "Geometry.h"

// Declaration
namespace isce {
    namespace geometry {
        // Main Topo class
        class Topo;
        // Some helper classes
        class Pixel;
        class Basis;
        class Layers;
    }
}

// Declare Topo class
class isce::geometry::Topo {

    public:
        // Constructor: must have Ellipsoid, Orbit, and Metadata
        inline Topo(isce::core::Ellipsoid,
                    isce::core::Orbit,
                    isce::core::Metadata);

        // Set options
        inline void initialized(bool);
        inline void threshold(double);
        inline void numiter(int);
        inline void extraiter(int);
        inline void orbitMethod(isce::core::orbitInterpMethod);
        inline void demMethod(isce::core::dataInterpMethod);

        // Check initialization
        inline void checkInitialization() const;

        // Run topo
        void topo(isce::core::Raster &,
                  isce::core::Poly2d &,
                  isce::core::Poly2d &,
                  const std::string);

    private:
        // isce::core objects
        isce::core::Orbit _orbit;
        isce::core::Ellipsoid _ellipsoid;
        isce::core::Metadata _meta;
        // Local isce::core peg objects
        isce::core::Peg _peg;
        isce::core::Pegtrans _ptm;
    
        // Optimization options
        double _threshold;
        int _numiter, _extraiter, _orbitMethod, _demMethod;
        // Flag to make sure options have been initialized
        bool _initialized;
};

class isce::geometry::Pixel {

    public:
        // Constructors
        Pixel() {};
        Pixel(double r, double d, size_t b) : _range(r), _dopfact(d) _bin(b) {}
        // Getters
        double range() const { return _range; }
        double dopfact() const { return _dopfact; }
        size_t bin() const { return _bin; }
        // Setters
        void range(double r) { _range = r; }
        void dopfact(double d) { _dopfact = d; }
        void bin(size_t b) { _bin = b; }

    private:
        double _range;
        double _dopfact;
        size_t _bin;
};

class isce::geometry::Basis {

    public:
        // Constructors
        Basis() {};
        Basis(cartesian_t & t, cartesian_t & c, cartesian_t & n) :
            _that(t), _chat(c), _nhat(n) {}
        // Getters
        cartesian_t that() const { return _that; }
        cartesian_t chat() const { return _chat; }
        cartesian_t nhat() const { return _nhat; }
        // Setters
        void that(cartesian_t & t) { _that = t; }
        void chat(cartesian_t & c) { _chat = t; }
        void nhat(cartesian_t & n) { _nhat = t; }

    private:
        cartesian_t _that;
        cartesian_t _chat;
        cartesian_t _nhat;
};

class isce::geometry::TopoLayers {

    public:
        // Constructors
        TopoLayers(size_t width) : _width(width) {
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
        const std::valarray<double> & lat() const { return _lat; }
        const std::valarray<double> & lon() const { return _lon; }
        const std::valarray<double> & z() const { return _z; }
        const std::valarray<float> & inc() const { return _inc; }
        const std::valarray<float> & hdg() const { return _hdg; }
        const std::valarray<float> & localInc() const { return _localInc; }
        const std::valarray<float> & localPsi() const { return _localPsi; }
        const std::valarray<float> & sim() const { return _sim; }

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
        // Size information
        size_t _width;
};

#endif

// end of file
