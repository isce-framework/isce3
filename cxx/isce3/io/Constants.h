//-*- C++ -*-

#pragma once

#include <string>
#include <unordered_map>
#include <gdal_priv.h>

#include "gdal/detail/GDALDataTypeUtil.h"

//! The isce namespace
namespace isce {
    //! The isce::io namespace
    namespace io {

        // Constants for Raster class
        /// Default GDAL driver used by Raster for creation
        const std::string defaultGDALDriver = "VRT";
        /// Default GDAL data type used by Raster for creation
        const GDALDataType defaultGDALDataType = GDT_Float32;

        template<typename T>
        auto asGDT = gdal::detail::Type2GDALDataType<T>::datatype;
    }
}
