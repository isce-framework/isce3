//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_CONSTANTS_H__
#define __ISCE_CORE_CONSTANTS_H__


#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#include <typeindex>
#include <complex>
#include <cstdint>
#include "gdal_priv.h"


// Macro wrapper to check vector lengths (adds calling function and variable name information to the
// exception)
#define checkVecLen(v,l) isce::core::checkVecLenDebug(v,l,#v,__PRETTY_FUNCTION__)
#define check2dVecLen(v,l,w) isce::core::check2dVecLenDebug(v,l,w,#v,__PRETTY_FUNCTION__)
// Macro wrapper to provide 2D indexing to a 1D array
#define IDX1D(i,j,w) (((i)*(w))+(j))

namespace isce { namespace core {
    
    // Useful typedefs for 3-element vectors and 2D matrices
    // Will be replaced by dedicated array library
    typedef std::array<double, 3> cartesian_t;
    typedef std::array<std::array<double, 3>, 3> cartmat_t;

    enum orbitConvMethod {
        SCH_2_XYZ,
        XYZ_2_SCH
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
    enum sincInterpConst {
        SINC_HALF = 4,
        SINC_LEN = 8,
        SINC_ONE = 9,
        SINC_SUB = 8192
    };

    enum dataInterpMethod {
        SINC_METHOD,
        BILINEAR_METHOD,
        BICUBIC_METHOD,
        NEAREST_METHOD,
        AKIMA_METHOD,
        BIQUINTIC_METHOD
    };

    // Ellipsoid parameters for Earth
    const double EarthSemiMajorAxis = 6378137.0;
    const double EarthEccentricitySquared = 0.0066943799901;

    // Inline function for input checking on vector lengths (primarily to check to see if 3D vector 
    // has the correct number of inputs, but is generalized to any length). 'vec_name' is passed in
    // by the wrapper macro (stringified vector name), and 'parent_func' is passed in the same way 
    // through __PRETTY_FUNCTION__
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

    // Constants for Raster class
    const std::string defaultGDALDriver = "VRT"; 
    const GDALDataType defaultGDALDataType = GDT_Float32;
    // Unordered_map to map typeids to GDALDataTypes
    const std::unordered_map<std::type_index, GDALDataType> GDT =
      {{typeid(uint8_t),               GDT_Byte},
       {typeid(uint16_t),              GDT_UInt16},
       {typeid(int16_t),               GDT_Int16},
       {typeid(uint32_t),              GDT_UInt32},
       {typeid(int32_t),               GDT_Int32},
       {typeid(float),                 GDT_Float32},
       {typeid(double),                GDT_Float64},
       {typeid(std::complex<int16_t>), GDT_CInt16},
       {typeid(std::complex<int32_t>), GDT_CInt32},
       {typeid(std::complex<float>),   GDT_CFloat32},
       {typeid(std::complex<double>),  GDT_CFloat64}};
  }
}

#endif
