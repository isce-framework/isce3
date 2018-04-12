//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_GEOMETRY_BASIS_H
#define ISCE_GEOMETRY_BASIS_H

// isce::core
#include <isce/core/Constants.h>


// Declaration
namespace isce {
    namespace geometry {
        // Expose some useful isce::core typedefs
        typedef isce::core::cartesian_t cartesian_t;
        typedef isce::core::cartmat_t cartmat_t;
        // Basis class
        class Basis;
    }
}

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
        void chat(cartesian_t & c) { _chat = c; }
        void nhat(cartesian_t & n) { _nhat = n; }

    private:
        cartesian_t _that;
        cartesian_t _chat;
        cartesian_t _nhat;
};
    
#endif

// end of file
