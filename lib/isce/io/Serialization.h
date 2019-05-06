//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

/** \file io/Serialization.h
 *
 * Serialization utilities using HDF5 API. */

#ifndef ISCE_IO_SERIALIZATION_H
#define ISCE_IO_SERIALIZATION_H

#include <iostream>
#include <sstream>

// isce::core
#include <isce/core/DateTime.h>

// isce::io
#include <isce/io/IH5.h>

//! The isce namespace
namespace isce {
    //! The isce::io namespace
    namespace io {

        /** Check existence of a dataset or group
         *
         * @param[in] h5obj         HDF5 file or group object.
         * @param[in] name          H5 path of dataset relative to h5obj. 
         * @param[in] start         Relative path from current group/file to start search from. 
         * @param[in] type          Type of object to search for. Default: BOTH */
        template <typename H5obj>
        inline bool exists(H5obj & h5obj, const std::string & name,
                           const std::string start = ".", const std::string type = "BOTH") {
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
        template <typename H5obj, typename T>
        inline void loadFromH5(H5obj & h5obj, const std::string & datasetPath, T & v) {
            // Open dataset
            isce::io::IDataSet dataset = h5obj.openDataSet(datasetPath);
            // Read the scalar dataset
            dataset.read(v);
        }

        /** Load vector dataset from HDF5 file.
         *
         * @param[in] file          HDF5 file or group object.
         * @param[in] datasetPath   H5 path of dataset relative to h5obj.
         * @param[in] v             Vector to store dataset. */
        template <typename H5obj, typename T>
        inline void loadFromH5(H5obj & h5obj, const std::string & datasetPath,
                               std::vector<T> & v) {
            // Open dataset
            isce::io::IDataSet dataset = h5obj.openDataSet(datasetPath);
            // Read the vector dataset
            dataset.read(v);
        }

        /** Load valarray dataset from HDF5 file.
         *
         * @param[in] file          HDF5 file or group object.
         * @param[in] datasetPath   H5 path of dataset relative to h5obj.
         * @param[in] v             Valarray to store dataset. */
        template <typename H5obj, typename T>
        inline void loadFromH5(H5obj & h5obj, const std::string & datasetPath,
                               std::valarray<T> & v) {
            // Open dataset
            isce::io::IDataSet dataset = h5obj.openDataSet(datasetPath);
            // Read the valarray dataset
            dataset.read(v);
        }

        /** Load Matrix dataset from HDF5 file.
          *
          * @param[in] file          HDF5 file or group object.
          * @param[in] datasetPath   H5 path of dataset relative to h5obj.
          * @param[in] m             Matrix to store dataset. */
        template <typename H5obj, typename T>
        inline void loadFromH5(H5obj & h5obj, const std::string & datasetPath,
                               isce::core::Matrix<T> & m) {
            // Open dataset
            isce::io::IDataSet dataset = h5obj.openDataSet(datasetPath);
            // Read the Matrix dataset using raw pointer interface
            dataset.read(m.data());
        }

        /** Write vector dataset to HDF5 file.
         *
         * @param[in] file          HDF5 file or group object.
         * @param[in] datasetPath   H5 path of dataset relative to h5obj.
         * @param[in] v             Vector to write.
         * @param[in] units         Units of dataset. */
        template <typename H5obj, typename T>
        inline void saveToH5(H5obj & h5obj, const std::string & datasetPath,
                             const std::vector<T> & v, const std::string & units = "") {
            // Check for existence of dataset
            if (exists(h5obj, datasetPath)) {
                return;
            }
            // Create dataset
            isce::io::IDataSet dset = h5obj.createDataSet(datasetPath, v);
            // Update units attribute if long enough
            if (units.length() > 0) {
                dset.createAttribute("units", units);
            }
        }

        /** Write valarray dataset to HDF5 file.
         *
         * @param[in] file          HDF5 file or group object.
         * @param[in] datasetPath   H5 path of dataset relative to h5obj.
         * @param[in] v             Valarray to write.
         * @param[in] units         Units of dataset. */
        template <typename H5obj, typename T>
        inline void saveToH5(H5obj & h5obj, const std::string & datasetPath,
                             const std::valarray<T> & v, const std::string & units = "") {
            // Check for existence of dataset
            if (exists(h5obj, datasetPath)) {
                return;
            }
            // Create dataset
            isce::io::IDataSet dset = h5obj.createDataSet(datasetPath, v);
            // Update units attribute if long enough
            if (units.length() > 0) {
                dset.createAttribute("units", units);
            }
        }

        /** Write vector dataset with dimensions to HDF5 file.
         *
         * @param[in] file          HDF5 file or group object.
         * @param[in] datasetPath   H5 path of dataset relative to h5obj.
         * @param[in] v             Vector to write.
         * @param[in] units         Units of dataset. */
        template <typename H5obj, typename T, size_t S>
        inline void saveToH5(H5obj & h5obj, const std::string & datasetPath,
                             const std::vector<T> & v, std::array<size_t, S> dims,
                             const std::string & units = "") {
            // Check for existence of dataset
            if (exists(h5obj, datasetPath)) {
                return;
            }
            // Create dataset
            isce::io::IDataSet dset = h5obj.createDataSet(datasetPath, v, dims);
            // Update units attribute if long enough
            if (units.length() > 0) {
                dset.createAttribute("units", units);
            }
        }

        /** Write valarray dataset with dimensions to HDF5 file.
         *
         * @param[in] file          HDF5 file or group object.
         * @param[in] datasetPath   H5 path of dataset relative to h5obj.
         * @param[in] v             Valarray to write.
         * @param[in] units         Units of dataset. */
        template <typename H5obj, typename T, size_t S>
        inline void saveToH5(H5obj & h5obj, const std::string & datasetPath,
                             const std::valarray<T> & v, std::array<size_t, S> dims,
                             const std::string & units = "") {
            // Check for existence of dataset
            if (exists(h5obj, datasetPath)) {
                return;
            }
            // Create dataset
            isce::io::IDataSet dset = h5obj.createDataSet(datasetPath, v, dims);
            // Update units attribute if long enough
            if (units.length() > 0) {
                dset.createAttribute("units", units);
            }
        }

        /** Write Matrix dataset to HDF5 file.
         *
         * @param[in] file          HDF5 file or group object.
         * @param[in] datasetPath   H5 path of dataset relative to h5obj.
         * @param[in] mat           Matrix to write.
         * @param[in] units         Units of dataset. */
        template <typename H5obj, typename T>
        inline void saveToH5(H5obj & h5obj, const std::string & datasetPath,
                             const isce::core::Matrix<T> & mat, 
                             const std::string & units = "") {
            // Check for existence of dataset
            if (exists(h5obj, datasetPath)) {
                return;
            }
            // Create dataset
            std::array<size_t, 2> dims{mat.length(), mat.width()};
            isce::io::IDataSet dset = h5obj.createDataSet(datasetPath, mat.data(), dims);
            // Update units attribute if long enough
            if (units.length() > 0) {
                dset.createAttribute("units", units);
            }
        }

        /** Get dimensions of complex imagery from HDF5 file.
         *
         * @param[in] file          HDF5 file or group object.
         * @param[in] datasetPath   H5 path of image dataset relative to h5obj. */
        template <typename H5obj>
        inline std::vector<int> getImageDims(H5obj & h5obj, const std::string & datasetPath) {
            // Open dataset
            isce::io::IDataSet dataset = h5obj.openDataSet(datasetPath);
            // Get dimensions
            return dataset.getDimensions();
        }

        /** Parse time units in a dataset attribute to get a reference epoch
          *
          * @param[in] file         HDF5 file or group object.
          * @param[in] datasetPath  H5 path of dataset relative to h5obj.
          * @param[out] epoch       isce::core::DateTime of reference epoch. */
        template <typename H5obj>
        inline isce::core::DateTime getRefEpoch(H5obj & h5obj, const std::string & datasetPath) {

            // Open the dataset
            isce::io::IDataSet dset = h5obj.openDataSet(datasetPath);

            // Get the attribute
            std::string unitAttr;
            std::string attribute("units");
            dset.read(unitAttr, attribute);

            // Parse it
            std::string dummy1, dummy2, datestr, timestr;
            std::istringstream ss(unitAttr);
            ss >> dummy1 >> dummy2 >> datestr >> timestr;
            std::string inputstr = datestr + "T" + timestr + ".00";
            isce::core::DateTime epoch(inputstr);

            // Done
            return epoch;
        }

        /** Save reference epoch DateTime as an attribute.
          *
          * @param[in] file         HDF5 file or group object.
          * @param[in] datasetPath  H5 path of dataset relative to h5obj.
          * @param[in] epoch        isce::core::DateTime of reference epoch. */
        template <typename H5obj>
        inline void setRefEpoch(H5obj & h5obj, const std::string & datasetPath,
                                const isce::core::DateTime & refEpoch) {

            // Open the dataset
            isce::io::IDataSet dset = h5obj.openDataSet(datasetPath);
            
            // Need to create string representation of DateTime manually
            char buffer[40];
            sprintf(buffer,
                    "seconds since %04d-%02d-%02d %02d:%02d:%02d",
                    refEpoch.year,
                    refEpoch.months,
                    refEpoch.days,
                    refEpoch.hours,
                    refEpoch.minutes,
                    refEpoch.seconds);
            std::string unitsAttr{buffer};

            // Save buffer to attribute
            dset.createAttribute("units", unitsAttr);
        }

    }
}

#endif

// end of file
