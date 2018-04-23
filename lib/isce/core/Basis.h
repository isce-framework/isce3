//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_CORE_BASIS_H
#define ISCE_CORE_BASIS_H

// isce::core
#include <isce/core/Constants.h>


// Declaration
namespace isce {
    namespace core {
        class Basis;
    }
}

class isce::core::Basis {

    public:
        // Constructors
        Basis() {};
        Basis(cartesian_t & x0, cartesian_t & x1, cartesian_t & x2) :
            _x0(x0), _x1(x1), _x2(x2) {}
        // Getters
        cartesian_t x0() const { return _x0; }
        cartesian_t x1() const { return _x1; }
        cartesian_t x2() const { return _x2; }
        // Setters
        void x0(cartesian_t & x0) { _x0 = x0; }
        void x1(cartesian_t & x1) { _x1 = x1; }
        void x2(cartesian_t & x2) { _x2 = x2; }

    private:
        cartesian_t _x0;
        cartesian_t _x1;
        cartesian_t _x2;
};
    
#endif

// end of file
