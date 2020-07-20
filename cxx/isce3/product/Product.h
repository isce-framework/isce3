// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018

#pragma once

// std
#include <string>
#include <algorithm>
#include <locale>
#include <map>

#include <isce3/core/Constants.h>
#include <isce3/core/LookSide.h>
#include <isce3/io/IH5.h>
#include <isce3/product/Metadata.h>
#include <isce3/product/Swath.h>

// Declarations
namespace isce3 {
    namespace product {
        class Product;
    }
}

// Product class declaration
class isce3::product::Product {

    public:
        /** Constructor from IH5File object. */
        Product(isce3::io::IH5File &);

        /** Constructor with Metadata and Swath map. */
        inline Product(const Metadata &, const std::map<char, isce3::product::Swath> &);

        /** Get a read-only reference to the metadata */
        inline const Metadata & metadata() const { return _metadata; }
        /** Get a reference to the metadata. */
        inline Metadata & metadata() { return _metadata; }

        /** Get a read-only reference to a swath */
        inline const Swath & swath(char freq) const { return _swaths.at(freq); }
        /** Get a reference to a swath */
        inline Swath & swath(char freq) { return _swaths[freq]; }
        /** Set a swath */
        inline void swath(const Swath & s, char freq) { _swaths[freq] = s; }

        /** Get the look direction */
        inline isce3::core::LookSide lookSide() const { return _lookSide; }
        /** Set look direction using enum */
        inline void lookSide(isce3::core::LookSide side) { _lookSide = side; }
        /** Set look direction from a string */
        inline void lookSide(const std::string &);

        /** Get the filename of the HDF5 file. */
        inline std::string filename() const { return _filename; }

    private:
        isce3::product::Metadata _metadata;
        std::map<char, isce3::product::Swath> _swaths;
        std::string _filename;
        isce3::core::LookSide _lookSide;
};

/** @param[in] meta Metadata object
  * @param[in] swaths Map of Swath objects per frequency */
isce3::product::Product::
Product(const Metadata & meta, const std::map<char, isce3::product::Swath> & swaths) :
    _metadata(meta), _swaths(swaths) {}

/** @param[in] look String representation of look side */
void
isce3::product::Product::
lookSide(const std::string & inputLook) {
    _lookSide = isce3::core::parseLookSide(inputLook);
}
