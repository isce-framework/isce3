//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

/** \file io/Serialization.h
 *
 * Serialization utilities using HDF5 API. */

#pragma once

#include <iostream>
#include <pyre/journal.h>
#include <sstream>

// isce3::core
#include <isce3/core/DateTime.h>

// isce3::io
#include <isce3/io/IH5.h>

//! The isce namespace
namespace isce3 {
//! The isce3::io namespace
namespace io {

/** Check existence of a dataset or group
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] name          H5 path of dataset relative to h5obj.
 * @param[in] start         Relative path from current group/file to start
 * search from.
 * @param[in] type          Type of object to search for. Default: BOTH */
template<typename H5obj>
inline bool exists(H5obj& h5obj, const std::string& name,
                   const std::string start = ".",
                   const std::string type = "BOTH")
{
    // Search for objects
    std::vector<std::string> objs = h5obj.find(name, start, type, "FULL");
    // Check size of vector
    if (objs.size() > 0) {
        return true;
    } else {
        return false;
    }
}

/** Load scalar dataset from HDF5 file.
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] datasetPath   H5 path of dataset relative to h5obj.
 * @param[in] v             Scalar return value. */
template<typename H5obj, typename T>
inline void loadFromH5(H5obj& h5obj, const std::string& datasetPath, T& v)
{
    // Open dataset
    isce3::io::IDataSet dataset = h5obj.openDataSet(datasetPath);
    // Read the scalar dataset
    dataset.read(v);
}

/** Load vector dataset from HDF5 file.
 *
 * @param[in] h5obj          HDF5 file or group object.
 * @param[in] datasetPath   H5 path of dataset relative to h5obj.
 * @param[in] v             Vector to store dataset. */
template<typename H5obj, typename T>
inline void loadFromH5(H5obj& h5obj, const std::string& datasetPath,
                       std::vector<T>& v)
{
    // Open dataset
    isce3::io::IDataSet dataset = h5obj.openDataSet(datasetPath);
    // Read the vector dataset
    dataset.read(v);
}

/** Load valarray dataset from HDF5 file.
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] datasetPath   H5 path of dataset relative to h5obj.
 * @param[in] v             Valarray to store dataset. */
template<typename H5obj, typename T>
inline void loadFromH5(H5obj& h5obj, const std::string& datasetPath,
                       std::valarray<T>& v)
{
    // Open dataset
    isce3::io::IDataSet dataset = h5obj.openDataSet(datasetPath);
    // Read the valarray dataset
    dataset.read(v);
}

/** Load Matrix dataset from HDF5 file.
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] datasetPath   H5 path of dataset relative to h5obj.
 * @param[in] m             Matrix to store dataset. */
template<typename H5obj, typename T>
inline void loadFromH5(H5obj& h5obj, const std::string& datasetPath,
                       isce3::core::Matrix<T>& m)
{
    // Open dataset
    isce3::io::IDataSet dataset = h5obj.openDataSet(datasetPath);
    // Read the Matrix dataset using raw pointer interface
    dataset.read(m.data());
}

/** Write scalar dataset to HDF5 file.
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] datasetPath   H5 path of dataset relative to h5obj.
 * @param[in] val           Value to write
 * @param[in] units         Units of dataset. */
template<typename H5obj, typename T>
inline void saveToH5(H5obj& h5obj, const std::string& datasetPath, const T& val,
                     const std::string& units = "")
{
    // Check for existence of dataset
    if (exists(h5obj, datasetPath)) {
        return;
    }
    // Create dataset
    isce3::io::IDataSet dset = h5obj.createDataSet(datasetPath, val);
    // Update units attribute if long enough
    if (units.length() > 0) {
        dset.createAttribute("units", units);
    }
}

/** Write vector dataset to HDF5 file.
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] datasetPath   H5 path of dataset relative to h5obj.
 * @param[in] v             Vector to write.
 * @param[in] units         Units of dataset. */
template<typename H5obj, typename T>
inline void saveToH5(H5obj& h5obj, const std::string& datasetPath,
                     const std::vector<T>& v, const std::string& units = "")
{
    // Check for existence of dataset
    if (exists(h5obj, datasetPath)) {
        return;
    }
    // Create dataset
    isce3::io::IDataSet dset = h5obj.createDataSet(datasetPath, v);
    // Update units attribute if long enough
    if (units.length() > 0) {
        dset.createAttribute("units", units);
    }
}

/** Write valarray dataset to HDF5 file.
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] datasetPath   H5 path of dataset relative to h5obj.
 * @param[in] v             Valarray to write.
 * @param[in] units         Units of dataset. */
template<typename H5obj, typename T>
inline void saveToH5(H5obj& h5obj, const std::string& datasetPath,
                     const std::valarray<T>& v, const std::string& units = "")
{
    // Check for existence of dataset
    if (exists(h5obj, datasetPath)) {
        return;
    }
    // Create dataset
    isce3::io::IDataSet dset = h5obj.createDataSet(datasetPath, v);
    // Update units attribute if long enough
    if (units.length() > 0) {
        dset.createAttribute("units", units);
    }
}

/** Write vector dataset with dimensions to HDF5 file.
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] datasetPath   H5 path of dataset relative to h5obj.
 * @param[in] v             Vector to write.
 * @param[in] units         Units of dataset. */
template<typename H5obj, typename T, size_t S>
inline void saveToH5(H5obj& h5obj, const std::string& datasetPath,
                     const std::vector<T>& v, std::array<size_t, S> dims,
                     const std::string& units = "")
{
    // Check for existence of dataset
    if (exists(h5obj, datasetPath)) {
        return;
    }
    // Create dataset
    isce3::io::IDataSet dset = h5obj.createDataSet(datasetPath, v, dims);
    // Update units attribute if long enough
    if (units.length() > 0) {
        dset.createAttribute("units", units);
    }
}

/** Write valarray dataset with dimensions to HDF5 file.
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] datasetPath   H5 path of dataset relative to h5obj.
 * @param[in] v             Valarray to write.
 * @param[in] units         Units of dataset. */
template<typename H5obj, typename T, size_t S>
inline void saveToH5(H5obj& h5obj, const std::string& datasetPath,
                     const std::valarray<T>& v, std::array<size_t, S> dims,
                     const std::string& units = "")
{
    // Check for existence of dataset
    if (exists(h5obj, datasetPath)) {
        return;
    }
    // Create dataset
    isce3::io::IDataSet dset = h5obj.createDataSet(datasetPath, v, dims);
    // Update units attribute if long enough
    if (units.length() > 0) {
        dset.createAttribute("units", units);
    }
}

/** Write Matrix dataset to HDF5 file.
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] datasetPath   H5 path of dataset relative to h5obj.
 * @param[in] mat           Matrix to write.
 * @param[in] units         Units of dataset. */
template<typename H5obj, typename T>
inline void saveToH5(H5obj& h5obj, const std::string& datasetPath,
                     const isce3::core::Matrix<T>& mat,
                     const std::string& units = "")
{
    // Check for existence of dataset
    if (exists(h5obj, datasetPath)) {
        return;
    }
    // Create dataset
    std::array<size_t, 2> dims {mat.length(), mat.width()};
    isce3::io::IDataSet dset =
            h5obj.createDataSet(datasetPath, mat.data(), dims);
    // Update units attribute if long enough
    if (units.length() > 0) {
        dset.createAttribute("units", units);
    }
}

/** Get dimensions of complex imagery from HDF5 file.
 *
 * @param[in] h5obj         HDF5 file or group object.
 * @param[in] datasetPath   H5 path of image dataset relative to h5obj. */
template<typename H5obj>
inline std::vector<int> getImageDims(H5obj& h5obj,
                                     const std::string& datasetPath)
{
    // Open dataset
    isce3::io::IDataSet dataset = h5obj.openDataSet(datasetPath);
    // Get dimensions
    return dataset.getDimensions();
}

/** Parse time units in a dataset attribute to get a reference epoch
 *
 * @param[in] h5obj        HDF5 file or group object.
 * @param[in] datasetPath  H5 path of dataset relative to h5obj.
 * @param[out] epoch       isce3::core::DateTime of reference epoch.
 * @exception InvalidArgument  if no matching iso8601 string is found.
 */

template<typename H5obj>
inline isce3::core::DateTime getRefEpoch(H5obj& h5obj,
                                         const std::string& datasetPath)
{

    // Open the dataset
    isce3::io::IDataSet dset = h5obj.openDataSet(datasetPath);

    // Get the attribute
    std::string unitAttr;
    std::string attribute("units");
    dset.read(unitAttr, attribute);

    // Parse it
    // find UTC date-time string
    std::string utc_ref {};
    const auto pos_iso {unitAttr.find_last_of('-')};
    if (pos_iso != std::string::npos) {
        utc_ref = unitAttr.substr(pos_iso - 7, 30);
        // get rid of any trailing white space
        auto pos_last_wspc = utc_ref.find_last_not_of(" ");
        if (pos_last_wspc != std::string::npos)
            utc_ref = utc_ref.substr(0, pos_last_wspc + 1);
    }
    isce3::core::DateTime epoch(utc_ref);
    // Done
    return epoch;
}

/** Save reference epoch DateTime as an attribute.
 *
 * @param[in] h5obj        HDF5 file or group object.
 * @param[in] datasetPath  H5 path of dataset relative to h5obj.
 * @param[in] epoch        isce3::core::DateTime of reference epoch. */
template<typename H5obj>
inline void setRefEpoch(H5obj& h5obj, const std::string& datasetPath,
                        const isce3::core::DateTime& refEpoch)
{
    if (refEpoch.frac != 0.0) {
        auto log = pyre::journal::error_t("isce.io.Serialization.setRefEpoch");
        log << pyre::journal::at(__HERE__) << "Truncating fractional part of "
            "reference epoch " << std::string(refEpoch) << " while serializing "
            "dataset " << datasetPath << " to HDF5" << pyre::journal::endl;
    }

    // Open the dataset
    isce3::io::IDataSet dset = h5obj.openDataSet(datasetPath);

    // Need to create string representation of DateTime manually
    char buffer[40];
    snprintf(buffer, 40, "seconds since %04d-%02d-%02dT%02d:%02d:%02d",
            refEpoch.year, refEpoch.months, refEpoch.days, refEpoch.hours,
            refEpoch.minutes, refEpoch.seconds);
    std::string unitsAttr {buffer};

    // Save buffer to attribute
    dset.createAttribute("units", unitsAttr);
}

} // namespace io
} // namespace isce3
