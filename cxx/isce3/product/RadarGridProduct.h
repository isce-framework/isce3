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

namespace isce3 { namespace product {

/** Find unique group path excluding repeated occurrences
*/


/**
 * Return the path to each child group of `group` that ends with the substring
 * `group_name`.
 * 
 * \param[in] group       Parent group
 * \param[in] group_name  Search string
 * \returns               List of child group paths
 */
std::vector<std::string> findGroupPath(
    isce3::io::IGroup& group, const std::string& group_name);

/**
 * Return grids or swaths group paths within the base_group.
 * Start by assigning an empty string to image_group_str in case
 * grids and swaths group are not found.
 * 
 * \param[in] file
 * \param[in] base_dir           Path to `base_group` object (e.g. '/science/')
 * \param[in] base_group         Base group
 * \param[in] key_vector         Vector containing possible image groups
 * (e.g., 'swaths', 'grids', or both) to look for
 * \param[out] image_group_str   Path to first image group found containing 
 * one of the `key_vector` keys (e.g., '/science/LSAR/RSLC/swaths')
 * \param[in] metadata_group_str Path to first metadata group found by
 * substituting `key` with `metadata` in `image_group_str`
 * (e.g., '/science/LSAR/RSLC/metadata')
 */
void setImageMetadataGroupStr(
        isce3::io::IH5File & file,
        std::string& base_dir,
        isce3::io::IGroup& base_group,
        std::vector<std::string>& key_vector,
        std::string &image_group_str,
        std::string &metadata_group_str);

/** RadarGridProduct class declaration
 * 
 * The Produt attribute Swaths map, i.e. _swaths, associates the
 * frequency (key) with the Swath object (value). The RadarGridProduct object 
 * is usually initiated with an empty map and the serialization of 
 * the SAR product is responsible for populating the Swath map
 * from the product's metadata.
 *
 */
class RadarGridProduct {

    public:
        /** Constructor from IH5File object. */
        RadarGridProduct(isce3::io::IH5File &);
        /** Constructor with Metadata and Swath map. */
        inline RadarGridProduct(const Metadata &, const std::map<char, isce3::product::Swath> &);

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
isce3::product::RadarGridProduct::
RadarGridProduct(const Metadata & meta, const std::map<char, isce3::product::Swath> & swaths) :
    _metadata(meta), _swaths(swaths) {}

/** @param[in] look String representation of look side */
void
isce3::product::RadarGridProduct::
lookSide(const std::string & inputLook) {
    _lookSide = isce3::core::parseLookSide(inputLook);
}

}}
