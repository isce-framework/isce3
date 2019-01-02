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
            // Read the vector dataset
            dataset.read(v);
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

    }
}

#endif

// end of file
