#pragma once

#include <cstdint>
#include <complex>
#include <gdal_priv.h>
#include <type_traits>

// forward declare thrust::complex
namespace thrust {
    template<typename> struct complex;
}

namespace isce3 { namespace io { namespace gdal { namespace detail {

template<GDALDataType DataType>
struct GDT {
    static constexpr GDALDataType datatype = DataType;
};

template<typename T>
struct Type2GDALDataType : public GDT<GDT_Unknown> {};

// char is always a single byte
template<> struct Type2GDALDataType<char>          : public GDT<GDT_Byte> {};
template<> struct Type2GDALDataType<unsigned char> : public GDT<GDT_Byte> {};

#if GDAL_VERSION_NUM >= 3070000 // (GDAL>=3.7)
template<> struct Type2GDALDataType<std::int8_t>   : public GDT<GDT_Int8> {};
#endif

// fixed-size integer types
template<> struct Type2GDALDataType<std::int16_t> : public GDT<GDT_Int16> {};
template<> struct Type2GDALDataType<std::int32_t> : public GDT<GDT_Int32> {};

// fixed-size unsigned integer types
template<> struct Type2GDALDataType<std::uint16_t> : public GDT<GDT_UInt16> {};
template<> struct Type2GDALDataType<std::uint32_t> : public GDT<GDT_UInt32> {};

#if GDAL_VERSION_NUM >= 3050000 // (GDAL>=3.5)
template<> struct Type2GDALDataType<std::uint64_t> : public GDT<GDT_UInt64> {};
template<> struct Type2GDALDataType<std::int64_t> : public GDT<GDT_Int64> {};
#endif

// floating point types
template<> struct Type2GDALDataType<float>  : public GDT<GDT_Float32> {};
template<> struct Type2GDALDataType<double> : public GDT<GDT_Float64> {};

// complex floating point types
template<> struct Type2GDALDataType<std::complex<float>>  : public GDT<GDT_CFloat32> {};
template<> struct Type2GDALDataType<std::complex<double>> : public GDT<GDT_CFloat64> {};

// thrust::complex floating point types
template<> struct Type2GDALDataType<thrust::complex<float>>  : public GDT<GDT_CFloat32> {};
template<> struct Type2GDALDataType<thrust::complex<double>> : public GDT<GDT_CFloat64> {};

constexpr
std::size_t getSize(GDALDataType datatype)
{
    switch (datatype) {
        case GDT_Byte     : return sizeof(unsigned char);
#if GDAL_VERSION_NUM >= 3070000 // (GDAL>=3.7)
        case GDT_Int8     : return sizeof(std::int8_t);
#endif
        case GDT_UInt16   : return sizeof(std::uint16_t);
        case GDT_Int16    : return sizeof(std::int16_t);
        case GDT_UInt32   : return sizeof(std::uint32_t);
        case GDT_Int32    : return sizeof(std::int32_t);
#if GDAL_VERSION_NUM >= 3050000 // (GDAL>=3.5)
        case GDT_UInt64   : return sizeof(std::uint64_t);
        case GDT_Int64    : return sizeof(std::int64_t);
#endif
        case GDT_Float32  : return sizeof(float);
        case GDT_Float64  : return sizeof(double);
        case GDT_CFloat32 : return sizeof(std::complex<float>);
        case GDT_CFloat64 : return sizeof(std::complex<double>);
        default           : return 0;
    }
}

}}}}
