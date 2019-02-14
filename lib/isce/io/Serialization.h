//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

/** \file Serialization.h
 *
 * Serialization utilities using HDF5 API. */

#ifndef ISCE_IO_SERIALIZATION_H
#define ISCE_IO_SERIALIZATION_H

#include <iostream>
#include <sstream>

// isce::io
#include <isce/io/IH5.h>

//! The isce namespace
namespace isce {
    //! The isce::io namespace
    namespace io {

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
            std::stringstream ss;
            ss >> dummy1 >> dummy2 >> datestr >> timestr;
            isce::core::DateTime epoch(datestr + " " + timestr);

            // Done
            return epoch;
        }
            
    }
}

#endif

// end of file
