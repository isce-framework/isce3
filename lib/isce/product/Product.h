// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018

#ifndef ISCE_PRODUCT_PRODUCT_H
#define ISCE_PRODUCT_PRODUCT_H

// std
#include <string>
#include <algorithm>
#include <locale>
#include <map>

// isce::core
#include <isce/core/Constants.h>

// isce::product
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
        inline int lookSide() const { return _lookSide; }
        /** Set look direction from an integer*/
        inline void lookSide(int side) { _lookSide = side; }
        /** Set look direction from a string */
        inline void lookSide(const std::string &);

        /** Get the filename of the HDF5 file. */
        inline std::string filename() const { return _filename; }

    private:
        isce::product::Metadata _metadata;
        std::map<char, isce::product::Swath> _swaths;
        std::string _filename;
        int _lookSide;
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
    // Get look direction
    std::string lookDir;
    isce::io::loadFromH5(file, "/science/LSAR/identification/lookDirection", lookDir);
    lookSide(lookDir);
    // Save the filename
    _filename = file.filename();
}

/** @param[in] meta Metadata object
  * @param[in] swaths Map of Swath objects per frequency */
isce::product::Product::
Product(const Metadata & meta, const std::map<char, isce::product::Swath> & swaths) :
    _metadata(meta), _swaths(swaths) {}

/** @param[in] look String representation of look side */
void
isce::product::Product::
lookSide(const std::string & inputLook) {
    // Convert to lowercase
    std::string look(inputLook);
    std::for_each(look.begin(), look.end(), [](char & c) {
		c = std::tolower(c);
	});
    // Validate look string before setting
    if (look.compare("right") == 0) {
        _lookSide = -1;
    } else if (look.compare("left") == 0) {
        _lookSide = 1;
    } else {
        pyre::journal::error_t error("isce.product.Product");
        error
            << pyre::journal::at(__HERE__)
            << "Could not successfully set look direction. Not 'right' or 'left'."
            << pyre::journal::endl;
    }
}

#endif

// end of file
