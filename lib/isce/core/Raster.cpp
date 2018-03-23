//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: ml
// Original code: Joshua Cohen
// Copyright 2018
//

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "gdal_priv.h"
#include "Raster.h"
    

// Unordered_map to map typeids to GDALDataTypes
const std::unordered_map<std::type_index, GDALDataType> isce::core::Raster::_gdts =
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


// Construct a Raster object referring to existing file fname.
isce::core::Raster::Raster( const std::string &fname, bool readOnly=true) {
  if (GetGDALDriverManager()->GetDriverByName("VRT") == nullptr) GDALAllRegister();
  _access  = (GDALAccess) readOnly;
  _dataset = static_cast<GDALDataset*>(GDALOpenShared(fname.data(), _access));
}

// Construct a Raster object referring to new file fname.
isce::core::Raster::Raster(const std::string &fname, size_t width, size_t length, size_t numBands, GDALDataType dtype, const std::string & driverName) {
  if (GetGDALDriverManager()->GetDriverByName("VRT") == nullptr) GDALAllRegister();
  GDALDriver * outputDriver = GetGDALDriverManager()->GetDriverByName(driverName.c_str());
  _access  = GA_Update;
  _dataset = outputDriver->Create (fname.c_str(), width, length, numBands, dtype, NULL);
}

// Construct a Raster object referring to new file fname assuming ENVI driver.
isce::core::Raster::Raster(const std::string &fname, size_t width, size_t length, size_t numBands, GDALDataType dtype) :
  isce::core::Raster(fname, width, length, numBands, dtype, "ENVI") {}

// Construct a Raster object referring to new file fname assuming ENVI driver and float dataype.
isce::core::Raster::Raster(const std::string &fname, size_t width, size_t length, size_t numBands) :
  isce::core::Raster(fname, width, length, numBands, GDT_Float32, "ENVI") {}

// Construct a Raster object referring to new file fname assuming ENVI driver and one band.
isce::core::Raster::Raster(const std::string &fname, size_t width, size_t length, GDALDataType dtype) :
  isce::core::Raster(fname, width, length, 1, dtype, "ENVI") {}

// Construct a Raster object referring to new file fname assuming ENVI driver, float dataype and one band.
isce::core::Raster::Raster(const std::string &fname, size_t width, size_t length) :
  isce::core::Raster(fname, width, length, 1, GDT_Float32, "ENVI") {}

// Construct a Raster object referring to new file fname assuming ENVI driver, float dataype and one band.
isce::core::Raster::Raster(const std::string &fname, const Raster &rast) :
  isce::core::Raster(fname, rast.width(), rast.length(), rast.numBands(), rast.dtype(), "ENVI") {}

// Copy constructor. It increments GDAL's reference counter after weak-copying the pointer
isce::core::Raster::Raster(const Raster &rast) :
  _dataset(rast._dataset), _access(rast._access) {
  _dataset->Reference();
}

// Destructor. When GDALOpenShared() is used the dataset is dereferenced and closed only if the referenced count is below 1.
isce::core::Raster::~Raster() {
  GDALClose( _dataset );
}

// Load a dataset on existing Raster object after releasing the old dataset
void isce::core::Raster::load(const std::string &fname, bool readOnly=true) {
  GDALClose( _dataset );
  _access  = (GDALAccess) readOnly;
  _dataset = static_cast<GDALDataset*>(GDALOpenShared(fname.data(), _access));
}
