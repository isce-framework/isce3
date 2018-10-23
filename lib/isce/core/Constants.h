//
// Author: Joshua Cohen, Bryan Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_CONSTANTS_H
#define ISCE_CORE_CONSTANTS_H

#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#include <typeindex>
#include <complex>
#include <cstdint>

// Macro wrapper to check vector lengths (adds calling function and variable name information to the
// exception)
#define checkVecLen(v,l) isce::core::checkVecLenDebug(v,l,#v,__PRETTY_FUNCTION__)
#define check2dVecLen(v,l,w) isce::core::check2dVecLenDebug(v,l,w,#v,__PRETTY_FUNCTION__)
// Macro wrapper to provide 2D indexing to a 1D array
#define IDX1D(i,j,w) (((i)*(w))+(j))

namespace isce { namespace core {
    
    // Useful typedefs for 3-element vectors and 2D matrices
    // Will be replaced by dedicated array library
    /**Common datatype for a triplet of doubles*/
    typedef std::array<double, 3> cartesian_t;

    /**Common datatype for collection of 3 triplets of doubles*/
    typedef std::array<std::array<double, 3>, 3> cartmat_t;

    /**Enumeration type to indicate coordinate system of orbits*/
    enum orbitType {
        WGS84_ORBIT,
        SCH_ORBIT
    };

    /**Enumeration type to indicate method to use for orbit interpolation*/
    enum orbitInterpMethod {
        HERMITE_METHOD,
        SCH_METHOD,
        LEGENDRE_METHOD
    };
    
    /**Enumeration type to indicate interpolation method*/
    enum dataInterpMethod {
        SINC_METHOD,
        BILINEAR_METHOD,
        BICUBIC_METHOD,
        NEAREST_METHOD,
        AKIMA_METHOD,
        BIQUINTIC_METHOD
    };

    /** Default sinc parameters */
    const int SINC_HALF = 4;
    const int SINC_LEN = 8;
    const int SINC_ONE = 9;
    const int SINC_SUB = 8192;
    
    /** Semi-major axis for WGS84 */
    const double EarthSemiMajorAxis = 6378137.0;

    /** Eccentricity^2 for WGS84 */
    const double EarthEccentricitySquared = 0.0066943799901;

    /** Speed of light */
    const double SPEED_OF_LIGHT = 299792458.0;

    /** Struct with fixed-length string for serialization */
    struct FixedString {
        char str[50];
    };

     /* Inline function for input checking on vector lengths (primarily to check to see if 3D vector 
     * has the correct number of inputs, but is generalized to any length). 'vec_name' is passed in
     * by the wrapper macro (stringified vector name), and 'parent_func' is passed in the same way 
     * through __PRETTY_FUNCTION__*/
    template<typename T>
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
    template<typename T>
    inline void check2dVecLenDebug(const std::vector<std::vector<T>> &vec, size_t len, size_t width, 
                                   const char *vec_name, const char *parent_func) {
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
    template<typename T>
    std::vector<T> arange(T low, T high, T increment) {
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
    template<typename T>
    std::vector<T> linspace(T low, T high, std::size_t number) {
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
    template<typename T>
    std::vector<T> logspace(T first, T last, std::size_t number, T base=10) {
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

  }
}

#endif
