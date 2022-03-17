// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018

#pragma once

// std
#include <string>
#include <map>

#include <isce3/core/Constants.h>
#include <isce3/core/LookSide.h>
#include <isce3/io/IH5.h>
#include <isce3/product/Metadata.h>
#include <isce3/product/Grid.h>

// Declarations
namespace isce3 {
    namespace product {
        class GeoGridProduct;
    }
}

/** GeoGridProduct class declaration
 * 
 * The L2Produt attribute Grids map, i.e. _grids, associates the
 * frequency (key) with the Grids object (value). The GeoGridProduct object 
 * is usually initiated with an empty map and the serialization of 
 * the GeoGridProduct is responsible for populating the Grid map
 * from the GeoGridProduct's metadata.
 */
class isce3::product::GeoGridProduct {

    public:
        /** Constructor from IH5File object. */
        GeoGridProduct(isce3::io::IH5File &);

        /** Constructor with Metadata and Grid map. */
        inline GeoGridProduct(const Metadata &, const std::map<char, isce3::product::Grid> &);

        /** Get a read-only reference to the metadata */
        inline const Metadata & metadata() const { return _metadata; }
        /** Get a reference to the metadata. */
        inline Metadata & metadata() { return _metadata; }

        /** Get a read-only reference to a grid */
        inline const Grid & grid(char freq) const { return _grids.at(freq); }
        /** Get a reference to a grid */
        inline Grid & grid(char freq) { return _grids[freq]; }
        /** Set a grid */
        inline void grid(const Grid & s, char freq) { _grids[freq] = s; }

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
        std::map<char, isce3::product::Grid> _grids;
        std::string _filename;
        isce3::core::LookSide _lookSide;
};

/** @param[in] meta Metadata object
  * @param[in] grids Map of grid objects per frequency */
isce3::product::GeoGridProduct::
GeoGridProduct(const Metadata & meta, const std::map<char, isce3::product::Grid> & grids) :
    _metadata(meta), _grids(grids) {}


/** @param[in] look String representation of look side */
void
isce3::product::GeoGridProduct::
lookSide(const std::string & inputLook) {
    _lookSide = isce3::core::parseLookSide(inputLook);
}
