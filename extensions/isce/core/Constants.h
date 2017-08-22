//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_CONSTANTS_H__
#define __ISCE_CORE_CONSTANTS_H__

#include <stdexcept>
#include <string>
#include <vector>

// Macro wrapper to check vector lengths (adds calling function and variable name information to
// the exception)
#define checkVecLen(v,l) isceLib::checkVecLenDebug(v,l,#v,__PRETTY_FUNCTION__)
#define check2dVecLen(v,l,w) isceLib::check2dVecLenDebug(v,l,w,#v,__PRETTY_FUNCTION__)
// Macro wrapper to provide 2D indexing to a 1D array
#define IDX1D(i,j,w) (((i)*(w))+(j))

namespace isce::core {
    enum orbitConvMethod {
        SCH_2_XYZ,
        XYZ_2_SCH
    };

    enum latLonConvMethod {
        LLH_2_XYZ,
        XYZ_2_LLH,
        XYZ_2_LLH_OLD
    };

    enum orbitType {
        WGS84_ORBIT,
        SCH_ORBIT
    };

    enum orbitInterpMethod {
        HERMITE_METHOD,
        SCH_METHOD,
        LEGENDRE_METHOD
    };

    // Sinc interpolation constants
    static const int SINC_LEN = 8;
    static const int SINC_HALF = 4;
    static const int SINC_ONE = 9;
    static const int SINC_SUB = 8192;

    enum dataInterpMethod {
        SINC_METHOD,
        BILINEAR_METHOD,
        BICUBIC_METHOD,
        NEAREST_METHOD,
        AKIMA_METHOD,
        BIQUINTIC_METHOD
    };

    // Inline function for input checking on vector lengths (primarily to check to see if 3D vector has the correct number
    // of inputs, but is generalized to any length). 'vec_name' is passed in by the wrapper macro (stringified vector name),
    // and 'parent_func' is passed in the same way through __PRETTY_FUNCTION__
    template<typename T>
    inline void checkVecLenDebug(std::vector<T> &vec, size_t len, const char *vec_name, const char *parent_func) {
        if (vec.size() != len) {
            std::string errstr = "In '" + parent_func + "': Vector '" + vec_name + "' has size " + std::to_string(vec.size()) + 
                                    ", expected size " + std::to_string(len) + ".";
            throw std::invalid_argument(errstr);
        }
    }

    // Same as above but for 2D vectors
    template<typename T>
    inline void check2dVecLenDebug(std::vector<std::vector<T>> &vec, size_t len, size_t width, const char *vec_name, const char *parent_func) {
        if ((vec.size() != len) && (vec[0].size() != width)) {
            std::string errstr = "In '" + parent_func + "': Vector '" + vec_name + "' has size (" + std::to_string(vec.size()) + " x " + 
                                    std::to_string(vec[0].size()) + "), expected size (" + std::to_string(len) + " x " + std::to_string(width) + ").";
            throw std::invalid_argument(errstr);
        }
    }
}

#endif

