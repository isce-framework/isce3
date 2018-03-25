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


// Construct a Raster object referring to existing file
isce::core::Raster::Raster(const std::string &fname,          // filename
			   GDALAccess access) {   // GA_ReadOnly or GA_Update

  GDALAllRegister();  // GDAL checks internally if drivers are already loaded
  dataset( static_cast<GDALDataset*>(GDALOpenShared(fname.c_str(), access)) );
}

// Construct a Raster object referring to existing file
isce::core::Raster::Raster(const std::string &fname) :
  isce::core::Raster(fname, GA_ReadOnly) {}

// Construct a Raster object referring to new file
isce::core::Raster::Raster(const std::string &fname,          // filename 
			   size_t width,                      // number of columns
			   size_t length,                     // number of lines
			   size_t numBands,                   // number of bands
			   GDALDataType dtype,                // band datatype
			   const std::string & driverName) {  // GDAL raster format

  GDALAllRegister();
  GDALDriver * outputDriver = GetGDALDriverManager()->GetDriverByName(driverName.c_str());
  dataset( outputDriver->Create (fname.c_str(), width, length, numBands, dtype, NULL) );
}

// Construct a Raster object referring to new file assuming default GDAL driver.
isce::core::Raster::Raster(const std::string &fname, size_t width, size_t length, size_t numBands, GDALDataType dtype) :
  isce::core::Raster(fname, width, length, numBands, dtype, isce::core::defaultGDALDriver) {}

// Construct a Raster object referring to new file assuming default GDAL driver and dataype.
isce::core::Raster::Raster(const std::string &fname, size_t width, size_t length, size_t numBands) :
  isce::core::Raster(fname, width, length, numBands, isce::core::defaultGDALDataType) {}

// Construct a Raster object referring to new file assuming default GDAL driver and band.
isce::core::Raster::Raster(const std::string &fname, size_t width, size_t length, GDALDataType dtype) :
  isce::core::Raster(fname, width, length, 1, dtype, isce::core::defaultGDALDriver) {}

// Construct a Raster object referring to new file assuming default GDAL driver, dataype and band.
isce::core::Raster::Raster(const std::string &fname, size_t width, size_t length) :
  isce::core::Raster(fname, width, length, 1) {}

// Construct a Raster object referring to new file assuming default GDAL driver, dataype and band.
isce::core::Raster::Raster(const std::string &fname, const Raster &rast) :
  isce::core::Raster(fname, rast.width(), rast.length(), rast.numBands(), rast.dtype()) {}

// Copy constructor. It increments GDAL's reference counter after weak-copying the pointer
isce::core::Raster::Raster(const Raster &rast) {
  dataset( rast._dataset );
  dataset()->Reference();
}


// Construct a Raster object referring to a VRT dataset with multiple bands from a vector
// of Raster objects. Input rasters with multiple bands are unfolded within the output raster
// ToDo: Add member function isce::core::Raster::mergeRaster and call that function from here
isce::core::Raster::Raster(const std::string& fname, const std::vector<Raster>& rastVec) {
  
  for (auto r : rastVec)  // all rasters must have the same size
    if ( !r.match(rastVec.front()) ) 
      throw std::length_error("In isce::core::Raster::Raster() - Rasters have different sizes.");
  
  GDALAllRegister();
  GDALDriver * outputDriver = GetGDALDriverManager()->GetDriverByName("VRT");
  dataset( outputDriver->Create (fname.c_str(),
				 rastVec.front().width(),
				 rastVec.front().length(),
				 0,    // bands will be added later
				 rastVec.front().dtype(),
				 NULL) );  

  for ( auto r : rastVec)                       // for each input Raster object in rastVec
    for ( size_t b=1; b<=r.numBands(); ++b ) {  // for each band in input Raster object
      dataset()->AddBand( r.dtype(b), NULL );   // add the band to the output VRT dataset
      GDALRasterBand *tmp_band = dataset()->GetRasterBand( numBands() ); 
      VRTAddSimpleSource(tmp_band,
			 r.dataset()->GetRasterBand(b),
			 0, 0,
			 r.width(), r.length(), 
			 0, 0,
			 r.width(), r.length(),
			 NULL, VRT_NODATA_UNSET);
    }
}



// Destructor. When GDALOpenShared() is used the dataset is dereferenced
// and closed only if the referenced count is less than 1.
isce::core::Raster::~Raster() {
  GDALClose( _dataset );
}


// Load a dataset on existing Raster object after releasing the old dataset
void isce::core::Raster::open(const std::string &fname,
			      GDALAccess access=GA_ReadOnly) {
  GDALClose( _dataset );
  dataset( static_cast<GDALDataset*>(GDALOpenShared(fname.c_str(), access)) );
}

// end of file
