//-*- C++ -*_
//-*- coding: utf-8 -*-
//
// Authors: Bryan Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_CORE_UTILITIES_H
#define ISCE_CORE_UTILITIES_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <complex>
#include <cstdint>
#include <gdal.h>
#include <gdal_priv.h>

// Macro wrappers to check vector lengths 
// (adds calling function and variable name information to the exception)
#define checkVecLen(v,l) isce::core::checkVecLenDebug(v,l,#v,__PRETTY_FUNCTION__)
#define check2dVecLen(v,l,w) isce::core::check2dVecLenDebug(v,l,w,#v,__PRETTY_FUNCTION__)

namespace isce { namespace core {

    /** Inline function for input checking on vector lengths (primarily to check to see 
      * if 3D vector has the correct number of inputs, but is generalized to any length).
      * 'vec_name' is passed in by the wrapper macro (stringified vector name), and
      * 'parent_func' is passed in the same way through __PRETTY_FUNCTION__ */
    template <typename T>
    inline void checkVecLenDebug(const std::vector<T> &vec, size_t len, const char *vec_name, 
                                 const char *parent_func) {
        if (vec.size() != len) {
            std::string errstr = "In '" + std::string(parent_func) + "': Vector '" + 
                                 std::string(vec_name) + "' has size " + 
                                 std::to_string(vec.size()) + ", expected size " + 
                                 std::to_string(len) + ".";
            throw std::invalid_argument(errstr);
        }
    }

    // Same as above but for 2D vectors
    template <typename T>
    inline void check2dVecLenDebug(const std::vector<std::vector<T>> &vec,
                                   size_t len, size_t width, const char *vec_name,
                                   const char *parent_func) {
        if ((vec.size() != len) && (vec[0].size() != width)) {
            std::string errstr = "In '" + std::string(parent_func) + "': Vector '" + 
                                 std::string(vec_name) + "' has size (" + 
                                 std::to_string(vec.size()) + " x " + 
                                 std::to_string(vec[0].size()) + "), expected size (" + 
                                 std::to_string(len) + " x " + std::to_string(width) + ").";
            throw std::invalid_argument(errstr);
        }
    }

    /** Function to return a Python style arange vector 
      * @param[in] low Starting value of vector
      * @param[in] high Ending value of vector (non-inclusive)
      * @param[in] increment Spacing between vector elements */
    template <typename T>
    inline std::vector<T> arange(T low, T high, T increment) {
        // Instantiate the vector
        std::vector<T> data;
        // Set the first value
        T current = low;
        // Loop over the increments and add to vector
        while (current < high) {
            data.push_back(current);
            current += increment;
        }
        // done
        return data;
    }

    /** Function to return a Matlab/Python style linspace vector 
      * @param[in] low Starting value of vector
      * @param[in] high Ending value of vector (inclusive)
      * @param[in] number Number of vector elements */
    template <typename T>
    inline std::vector<T> linspace(T low, T high, std::size_t number) {
        // Instantiate the vector
        std::vector<T> data;
        // Compute the increment
        T increment = (high - low) / (number - 1);
        // Handle cases where number in (0, 1)
        if (number == 0) {
            return data;
        }
        if (number == 1) {
            data.push_back(low);
            return data;
        }
        // Loop over the increments and add to vector
        for (std::size_t i = 0; i < number - 1; ++i) {
            data.push_back(low + i * increment);
        }
        // Add the last element
        data.push_back(high);
        // done
        return data;
    }

    /** Function to return a Matlab/Python style logspace vector 
      * @param[in] first base^first is the starting value of the vector
      * @param[in] last base^last is the ending value of the vector
      * @param[in] Number of vector elements
      * @param[in] base Base of the log space */
    template <typename T>
    inline std::vector<T> logspace(T first, T last, std::size_t number, T base=10) {
        // Instantiate the vector
        std::vector<T> data;
        // Compute the increment of the powers
        T increment = (last - first) / (number - 1);
        // Handle cases where number in (0, 1)
        if (number == 0) {
            return data;
        }
        if (number == 1) {
            data.push_back(pow(base, first));
            return data;
        }
        // Loop over the increments and add to vector
        for (std::size_t i = 0; i < number - 1; ++i) {
            T value = pow(base, first + i * increment);
            data.push_back(value);
        }
        // Add the last element
        data.push_back(pow(base, last));
        // done
        return data;
    }

    /** Parse a space-separated string into a vector
      * @param[in] str String of values
      * @param[out] vec Vector of values */
    template <typename T>
    inline std::vector<T> stringToVector(const std::string & str) {
        // Make a stream
        std::stringstream stream(str);
        // Parse string
        T number;
        std::vector<T> vec;
        while (stream >> number) {
            vec.push_back(number);
        }
        // done
        return vec;
    }

    /** Generate a string of space-separated values from a vector
      * @param[in] vec Vector of values
      * @param[out] str String of values */
    template <typename T>
    inline std::string vectorToString(const std::vector<T> & vec) {
        // Initialize stream
        std::stringstream stream;
        // Print values
        std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(stream, " "));
        // done
        return stream.str();
    }

    /** Read metadata from a VRT file and return an std::string
      * @param[in] filename VRT product filename
      * @param[in] bandNum Band number to retrieve metadata from 
      * @param[out] meta std::string containing metadata */
    inline std::string stringFromVRT(const char * filename, int bandNum=1) {

        // Register GDAL drivers
        GDALAllRegister();

        // Open the VRT dataset
        GDALDataset * dataset = (GDALDataset *) GDALOpen(filename, GA_ReadOnly);
        if (dataset == NULL) {
            std::cout << "Cannot open dataset " << filename << std::endl;
            exit(1);
        }

        // Read the metadata
        char **metadata_str = dataset->GetRasterBand(bandNum)->GetMetadata("xml:isce");

        // The cereal-relevant XML is the first element in the list
        std::string meta{metadata_str[0]};

        // Close the VRT dataset
        GDALClose(dataset);

        // All done
        return meta;
    }

}}

#endif

// end of file
