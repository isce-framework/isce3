//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen
// Update: ml
// Copyright 2018
//

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gdal_priv.h"
#include "Raster.h"
    
// Define the GDALDataType mappings
const std::unordered_map<std::type_index,GDALDataType> isce::core::Raster::_gdts =
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

isce::core::Raster::Raster() : dataset(nullptr), _readonly(true) {
  /*
   *  Empty constructor also handles initializing the GDAL Drivers (if needed).
   */
  // Check if we've already registered the drivers by trying to load the VRT driver (the choice of
  // driver is arbitrary)
  if (GetGDALDriverManager()->GetDriverByName("VRT") == nullptr) {
    GDALAllRegister();
  }
}

isce::core::Raster::Raster(const std::string &fname, bool readOnly=true) : _readonly(readOnly) {
  /*
   *  Construct a Raster object referring to a particular image file, so load the dataset now.
   */
  if (GetGDALDriverManager()->GetDriverByName("VRT") == nullptr) {
    GDALAllRegister();
  }
  if (readOnly) dataset = static_cast<GDALDataset*>(GDALOpen(fname.data(), GA_ReadOnly));
  else dataset = static_cast<GDALDataset*>(GDALOpen(fname.data(), GA_Update));
  // Note that in most cases if GDALOpen fails it will throw a CPL_ERROR anyways
  if (dataset == nullptr) {
    std::string errstr = "In isce::core::Raster::Raster() - Cannot open file '";
    errstr += fname;
    errstr += "'.";
    throw std::runtime_error(errstr);
  }
}

isce::core::Raster::Raster(const Raster &rast) : dataset(rast.dataset), _readonly(rast._readonly) {
  /*
   *  Slightly enhanced copy constructor since we can't simply weak-copy the GDALDataset* pointer.
   *  Increment the reference to the GDALDataset's internal reference counter after weak-copying
   *  the pointer.
   */
  if (dataset) dataset->Reference();
}

isce::core::Raster::~Raster() {
  /*
   *  To account for the fact that multiple Raster objects might reference the same dataset, and
   *  to avoid duplicating the resources GDAL allocates under the hood, we work with GDAL's 
   *  reference/release system to handle GDALDataset* management. If you call release and the
   *  dataset's reference counter is 1, it automatically deletes the dataset.
   */
  if (dataset) dataset->Release();
}

void isce::core::Raster::loadFrom(const std::string &fname, bool readOnly=true) {
  /*
   *  Provides the same functionality as isce::core::Raster::Raster(std::string&,bool), but assumes that the
   *  GDALDrivers have already been initialized (since this can't be called as a static method).
   *  Will also try to delete any existing dataset that has been previously loaded (releasing
   *  stored resources).
   */
  // Dereference the currently-loaded GDALDataset (safer way to 'delete' since the loaded
  // GDALDataset might have come from copy-construction
  dataset->Release();
  if (readOnly) dataset = static_cast<GDALDataset*>(GDALOpen(fname.data(), GA_ReadOnly));
  else dataset = static_cast<GDALDataset*>(GDALOpen(fname.data(), GA_Update));
  _readonly = readOnly;
  // Note that in most cases if GDALOpen fails it will throw a CPL_ERROR anyways
  if (dataset == nullptr) {
    std::string errstr = "In isce::core::Raster::loadFrom() - Cannot open file '";
    errstr += fname;
    errstr += "'.";
    throw std::runtime_error(errstr);
  }
}
