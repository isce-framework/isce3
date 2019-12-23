#pragma once

#include <cstdint>
#include <complex>
#include <gdal_priv.h>
#include <type_traits>

// forward declare thrust::complex
namespace thrust {
    template<typename> struct complex;
}

namespace isce { namespace io { namespace gdal { namespace detail {

template<GDALDataType DataType>
struct GDT {
    static constexpr GDALDataType datatype = DataType;
};

template<typename T>
struct Type2GDALDataType : public GDT<GDT_Unknown> {};

// char is always a single byte
template<> struct Type2GDALDataType<char>          : public GDT<GDT_Byte> {};
template<> struct Type2GDALDataType<signed char>   : public GDT<GDT_Byte> {};
template<> struct Type2GDALDataType<unsigned char> : public GDT<GDT_Byte> {};

// fixed-size integer types
template<> struct Type2GDALDataType<std::int16_t> : public GDT<GDT_Int16> {};
template<> struct Type2GDALDataType<std::int32_t> : public GDT<GDT_Int32> {};

// fixed-size unsigned integer types
template<> struct Type2GDALDataType<std::uint16_t> : public GDT<GDT_UInt16> {};
template<> struct Type2GDALDataType<std::uint32_t> : public GDT<GDT_UInt32> {};

// floating point types
template<> struct Type2GDALDataType<float>  : public GDT<GDT_Float32> {};
template<> struct Type2GDALDataType<double> : public GDT<GDT_Float64> {};

// complex floating point types
template<> struct Type2GDALDataType<std::complex<float>>  : public GDT<GDT_CFloat32> {};
template<> struct Type2GDALDataType<std::complex<double>> : public GDT<GDT_CFloat64> {};

// thrust::complex floating point types
template<> struct Type2GDALDataType<thrust::complex<float>>  : public GDT<GDT_CFloat32> {};
template<> struct Type2GDALDataType<thrust::complex<double>> : public GDT<GDT_CFloat64> {};

}}}}
