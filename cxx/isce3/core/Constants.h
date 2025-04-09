//
// Author: Joshua Cohen, Bryan Riel
// Copyright 2017-2018
//

#pragma once

#include <complex>
#include <cstdint>
#include <string>

// Macro wrapper to check vector lengths (adds calling function and variable name information to the
// exception)
#define checkVecLen(v,l) isce3::core::checkVecLenDebug(v,l,#v,__PRETTY_FUNCTION__)
#define check2dVecLen(v,l,w) isce3::core::check2dVecLenDebug(v,l,w,#v,__PRETTY_FUNCTION__)

// Macro wrapper to provide 2D indexing to a 1D array
#define IDX1D(i,j,w) (((i)*(w))+(j))

namespace isce3 { namespace core {

/**Enumeration type to indicate interpolation method*/
enum dataInterpMethod {
    SINC_METHOD = 0,
    BILINEAR_METHOD = 1,
    BICUBIC_METHOD = 2,
    NEAREST_METHOD = 3,
    BIQUINTIC_METHOD = 4
};

/** Default sinc parameters */
const int SINC_HALF = 4;
const int SINC_LEN = 8;
const int SINC_ONE = 9;
const int SINC_SUB = 8192;

/** Convert string to dataInterpMethod */
dataInterpMethod parseDataInterpMethod(const std::string & method);

/** Semi-major axis for WGS84 */
const double EarthSemiMajorAxis = 6378137.0;

/** Eccentricity^2 for WGS84 derived from inverse flattening 298.257223563 */
const double EarthEccentricitySquared = 0.006694379990141317;

/** Spin rate of Earth used for ECI/ECEF conversions, rad/s */
constexpr static double EarthSpinRate = 7.292115024135738e-5;

/** Speed of light in vacuum (m/s) */
constexpr static double speed_of_light = 299792458.0;

/** Global minimum height */
const double GLOBAL_MIN_HEIGHT = -500.0;

/** Global maximum height */
const double GLOBAL_MAX_HEIGHT = 9000.0;

/** Layover and shadow values */
const short SHADOW_VALUE = 1;
const short LAYOVER_VALUE = 2;
const short LAYOVER_AND_SHADOW_VALUE = 3;

/** Area projection algorithm radar grid margin **/
const int AREA_PROJECTION_RADAR_GRID_MARGIN = 100;
constexpr static float AREA_PROJECTION_MIN_VALID_SAMPLES_RATIO = 0.0;

/** Convert decimal degrees to meters approximately */
double inline decimaldeg2meters(double deg) { return deg * (M_PI/180.0) * 6.37e6; }

/** Precision-promotion to double/complex\<double\>  **/
template<typename T> struct double_promote;

/** Template specialization for float */
template<> struct double_promote<float>  { using type = double; };

/** Template specialization for double */
template<> struct double_promote<double> { using type = double; };

/** Template specialization for complex\<float\> */
template<> struct double_promote<std::complex<float>>  { using type = std::complex<double>; };

/** Template specialization for complex\<double\> */
template<> struct double_promote<std::complex<double>> { using type = std::complex<double>; };

}}
