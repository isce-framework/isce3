//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Marco Lavalle
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
    


// Construct a Raster object referring to existing file
isce::core::Raster::Raster(const std::string &fname,   // filename
			   GDALAccess access) {        // GA_ReadOnly or GA_Update

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
isce::core::Raster::Raster(const std::string& fname, const std::vector<Raster>& rastVec) {  
  GDALAllRegister();
  GDALDriver * outputDriver = GetGDALDriverManager()->GetDriverByName("VRT");
  dataset( outputDriver->Create (fname.c_str(),
				 rastVec.front().width(),
				 rastVec.front().length(),
				 0,    // bands are added below
				 rastVec.front().dtype(),
				 NULL) );  

  for ( auto r : rastVec)     // for each input Raster object in rastVec
    addRasterToVRT( r );
}



// Destructor. When GDALOpenShared() is used the dataset is dereferenced
// and closed only if the referenced count is less than 1.
isce::core::Raster::~Raster() {
  GDALClose( _dataset );
}


// end of file
