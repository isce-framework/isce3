// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018

#ifndef ISCE_PRODUCT_COMPLEXIMAGERY_H
#define ISCE_PRODUCT_COMPLEXIMAGERY_H

// isce::product
#include <isce/product/ImageMode.h>

// Declaration
namespace isce {
    namespace product {
        class ComplexImagery;
    }
}

// ComplexImagery class declaration
class isce::product::ComplexImagery {

    public:
        /** Default constructor. */
        ComplexImagery() {}

        /** Assignment operator. */
        ComplexImagery & operator=(const ComplexImagery &);

        /** Get a copy of the auxiliary mode. */
        ImageMode auxMode() const { return _auxMode; }
        /** Set auxiliary mode. */
        void auxMode(ImageMode & mode) { _auxMode = mode; }

        /** Get a copy of the primary mode. */
        ImageMode primaryMode() const { return _primaryMode; }
        /** Set primary mode. */
        void primaryMode(ImageMode & mode) { _primaryMode = mode; }

    private:
        // ImageMode data
        ImageMode _auxMode;
        ImageMode _primaryMode;
};

#endif

// end of file
