// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018

#ifndef ISCE_PRODUCT_PRODUCT_H
#define ISCE_PRODUCT_PRODUCT_H

// isce::product
#include <isce/product/ComplexImagery.h>
#include <isce/product/Metadata.h>

// Declarations
namespace isce {
    namespace product {
        class Product;
    }
}

// Product class declaration
class isce::product::Product {

    public:
        /** Default constructor. */
        Product() {};

        /** Get a reference to the complex imagery. */
        ComplexImagery & complexImagery() { return _complexImagery; }

        /** Get a reference to the metadata. */
        Metadata & metadata() { return _metadata; }

    private:
        ComplexImagery _complexImagery;
        Metadata _metadata;
};

#endif

// end of file
