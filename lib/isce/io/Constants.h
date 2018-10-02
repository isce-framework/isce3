//-*- C++ -*-

#ifndef ISCE_IO_CONSTANTS_H
#define ISCE_IO_CONSTANTS_H

#include <string>
#include <unordered_map>
#include <gdal_priv.h>

//! The isce namespace
namespace isce {
    //! The isce::io namespace
    namespace io {

        // Constants for Raster class
        /// Default GDAL driver used by Raster for creation
        const std::string defaultGDALDriver = "VRT";
        /// Default GDAL data type used by Raster for creation
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

// end of file
