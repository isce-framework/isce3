// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018

#ifndef ISCE_PRODUCT_PRODUCT_H
#define ISCE_PRODUCT_PRODUCT_H

// std
#include <string>
#include <map>

// isce::product
#include <isce/product/ComplexImagery.h>
#include <isce/product/Metadata.h>
#include <isce/product/Serialization.h>

// Declarations
namespace isce {
    namespace product {
        class Product;
    }
}

// Product class declaration
class isce::product::Product {

    public:
        /** Constructor from IH5File object. */
        inline Product(isce::io::IH5File &);

        /** Constructor with Metadata and Swath map. */
        inline Product(const Metadata &, const std::map<char, isce::product::Swath> &);

        /** Get a reference to the metadata. */
        inline Metadata & metadata() { return _metadata; }

        /** Get a reference to a swath */
        inline Swath & swath(char freq) { return _swaths[freq]; }
        /** Set a swath */
        inline void swath(const Swath & s, char freq) { _swaths[freq] = s; }

        /** Get the filename of the HDF5 file. */
        inline std::string filename() const { return _filename; }

    private:
        isce::product::Metadata _metadata;
        std::map<char, isce::product::Swath> _swaths;
        std::string _filename;
};

/** @param[in] file IH5File object for product. */
isce::product::Product::
Product(isce::io::IH5File & file) {
    // Get swaths group
    isce::io::IGroup imGroup = file.openGroup("/science/LSAR/SLC/swaths");
    // Configure swaths
    loadFromH5(imGroup, _swaths);
    // Get metadata group
    isce::io::IGroup metaGroup = file.openGroup("/science/LSAR/SLC/metadata"); 
    // Configure metadata
    loadFromH5(metaGroup, _metadata);
    // Save the filename
    _filename = file.filename();
}

/** @param[in] meta Metadata object
  * @param[in] swaths Map of Swath objects per frequency */
isce::product::Product::
Product(const Metadata & meta, const std::map<char, isce::product::Swath> & swaths) :
    _metadata(meta), _swaths(swaths) {}

#endif

// end of file
