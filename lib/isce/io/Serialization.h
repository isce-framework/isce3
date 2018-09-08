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

        /** Load scalar dataset from HDF5.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] datasetPath   H5 path of dataset.
         * @param[in] v             Scalar return value. */
        template <typename T>
        inline void loadFromH5(isce::io::IH5File & file, const std::string & datasetPath, T & v) {
            // Open dataset
            isce::io::IDataSet dataset = file.openDataSet(datasetPath);
            // Read the scalar dataset
            dataset.read(v);
        }

        /** Load vector dataset from HDF5.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] datasetPath   H5 path of dataset.
         * @param[in] v             Vector to store dataset. */
        template <typename T>
        inline void loadFromH5(isce::io::IH5File & file, const std::string & datasetPath,
                        std::vector<T> & v) {
            // Open dataset
            isce::io::IDataSet dataset = file.openDataSet(datasetPath);
            // Read the vector dataset
            dataset.read(v);
        }

        /** Get dimensions of complex imagery from HDF5.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] datasetPath   H5 path of image dataset. */
        inline std::vector<int> getImageDims(isce::io::IH5File & file,
                                      const std::string & datasetPath) {
            // Open dataset
            isce::io::IDataSet dataset = file.openDataSet(datasetPath);
            // Get dimensions
            return dataset.getDimensions();
        }

    }
}

#endif

// end of file
