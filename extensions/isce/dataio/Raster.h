//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_DATAIO_RASTER_H__
#define __ISCE_DATAIO_RASTER_H__

#include <complex>
#include <cstdint>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include "gdal_priv.h"

namespace isce { namespace dataio {
    struct Raster {
        // Since we need to pass an output datatype to GDAL's RasterIO using the GDALDataTypes,
        // we can use an unordered_map to map typeids to GDALDataTypes so that we can use RTTI to
        // convert on the fly. Note that this assumes the user's system hasn't overridden the
        // standard float type to mean the same as double (since there's no standard float32_t/
        // float64_t equivalent). Note that due to the way the compiler builds static types, this
        // needs to be declared here but defined as an inline-like implementation below.
        static const std::unordered_map<std::type_index,GDALDataType> _gdts;
        GDALDataset *_dataset;
        size_t _linecount;
        bool _readonly;

        Raster();
        Raster(const std::string&,bool);
        // Prevent copy/etc constructors because we would need to do a deep-copy of the _dataset
        // (since the ~Raster destructor explicitly closes/deletes the _dataset)
        Raster(const Raster&);
        inline Raster& operator=(const Raster&);
        ~Raster();

        void loadFrom(const std::string&,bool);
        inline size_t getLength() { return _dataset ? _dataset->GetRasterXSize() : 0; }
        inline size_t getWidth() { return _dataset ? _dataset->GetRasterYSize() : 0; }
        inline size_t getNumBands() { return _dataset ? _dataset->GetRasterCount() : 0; }
        inline void resetLineCounter() { _linecount = 0; }
        inline std::string getSourceDataType(size_t);

        template<typename T> void getSetValue(T&,size_t,size_t,size_t,bool);
        template<typename T> void getValue(T&,size_t,size_t,size_t);
        template<typename T> void getValue(T&,size_t,size_t);
        template<typename T> void setValue(T&,size_t,size_t,size_t);
        template<typename T> void setValue(T&,size_t,size_t);
        template<typename T> void getSetLine(T*,size_t,size_t,size_t,bool);
        template<typename T> void getLine(std::vector<T>&,size_t,size_t);
        template<typename T> void getLine(std::vector<T>&,size_t);
        template<typename T> void setLine(std::vector<T>&,size_t,size_t);
        template<typename T> void setLine(std::vector<T>&,size_t);
        template<typename T> void getLineSequential(std::vector<T>&,size_t);
        template<typename T> void getLineSequential(std::vector<T>&);
        template<typename T> void setLineSequential(std::vector<T>&,size_t);
        template<typename T> void setLineSequential(std::vector<T>&);
    };

    // Define the GDALDataType mappings
    const std::unordered_map<std::type_index,GDALDataType> Raster::_gdts =
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

    inline Raster& Raster::operator=(const Raster &rhs) {
        _dataset = rhs._dataset;
        // Since we're sharing the dataset between objects, and only weak-copying the pointer,
        // increment the GDALDataset reference counter appropriately
        _dataset->Reference();
        _linecount = rhs._linecount;
        _readonly = rhs._readonly;
        return *this;
    }

    inline std::string Raster::getSourceDataType(size_t band=1) {
        if (_dataset && (band <= getNumBands())) {
            return GDALGetDataTypeName(_dataset->GetRasterBand(band)->GetRasterDataType());
        } else {
            return "";
        }
    }
}}

#endif
