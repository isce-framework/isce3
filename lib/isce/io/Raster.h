//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Marco Lavalle
// Original code: Joshua Cohen
// Copyright 2018
//

#ifndef __ISCE_IO_RASTER_H__
#define __ISCE_IO_RASTER_H__

#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include <valarray>
#include "gdal_priv.h"
#include "gdal_vrt.h"
#include "ogr_spatialref.h"
#include "Constants.h"
#include "isce/core/Matrix.h"

//#include <pyre/journal.h>

namespace isce {
    namespace io {
        class Raster;
    }
}

/** Data structure meant to handle Raster I/O operations.
*
* This is currently a thin wrapper over GDAL's Dataset class with some simpler
* interfaces for I/O. ISCE is expected to only support North-up and West-left
* oriented rasters. */
class isce::io::Raster {

  public:

      Raster() {_dataset = nullptr;};

      /** Constructor to open an existing file with specified Access mode - defaults to read-only */
      Raster(const std::string& fname, GDALAccess access = GA_ReadOnly);

      /** Constructor to create a dataset */
      Raster(const std::string& fname, size_t width, size_t length,
             size_t numBands, GDALDataType dtype = isce::io::defaultGDALDataType,
             const std::string& driverName = isce::io::defaultGDALDriver);

      /** Constructor to create a 1 band dataset with default Driver */
      Raster(const std::string& fname, size_t width, size_t length, GDALDataType dtype = isce::io::defaultGDALDataType);

      /** Constructor for a 1 band dataset from isce::core::Matrix<T> */
      template<typename T> Raster(isce::core::Matrix<T> &matrix);

      // Constructor for a 1 band dataset from isce::core::Matrix<T>::view_type
      template<typename T> Raster(pyre::grid::View<T> &view);

      /** Create new raster object like another */
      Raster(const std::string& fname, const Raster& rast);
      template<typename T> Raster(const std::string& fname, const std::vector<T>& buffer,   size_t length) :
            Raster(fname, buffer.size(), length, 1, GDT.at(typeid(T))) {}
      template<typename T> Raster(const std::string& fname, const std::valarray<T>& buffer, size_t length) :
            Raster(fname, buffer.size(), length, 1, GDT.at(typeid(T))) {}

      /** Create a VRT raster dataset with collection of bands from Rasters */
      Raster(const std::string& fname, const std::vector<Raster>& rastVec);

      /** Copy constructor*/
      Raster(const Raster&);

      /** Constructor from an existing GDAL Dataset*/
      Raster(GDALDataset *inputDataset);

      /** Construct dataset for a 1 band dataset with raw pointer, dimensions and offsets */
      inline void initFromPointer(void* ptr, GDALDataType dtype, size_t width, size_t length, size_t pixeloffset, size_t lineoffset);

      /** Destructor */
      ~Raster();

      /** Assignment operator */
      inline Raster&      operator=(const Raster&);

      /** Length getter */
      inline size_t       length()   const { return _dataset->GetRasterYSize(); }

      /** Width getter */
      inline size_t       width()    const { return _dataset->GetRasterXSize(); }

      /** Number of bands getter*/
      inline size_t       numBands() const { return _dataset->GetRasterCount(); }

      /** Access mode getter */
      inline GDALAccess   access()   const { return _dataset->GetAccess(); }

      /** GDALDataset pointer getter */
      inline GDALDataset* dataset()  const { return _dataset; }

      /** GDALDataset pointer setter
       *
       * @param[in] ds GDALDataset pointer*/
      inline void         dataset(GDALDataset* ds) { _dataset=ds; }

      /** Return GDALDatatype of specified band
       *
       * @param[in] band Band number in 1-index*/
      inline GDALDataType dtype(const size_t band=1) const { return _dataset->GetRasterBand(band)->GetRasterDataType(); }

      /** Check dimensions compatibility with another raster
       *
       * @param[in] rast Reference raster to compare against*/
      inline bool         match(const Raster & rast) const { return width()==rast.width() && length()==rast.length(); }

      /**Open file with GDAL*/
      inline void         open(const std::string &fname, GDALAccess access);

      /** Add a raster to VRT*/
      inline void         addRasterToVRT(const isce::io::Raster& rast);

      /** Add a GDALRasterBand to VRT */
      inline void         addBandToVRT(GDALRasterBand *inBand);

      /** Add a raw data band to VRT */
      inline void         addRawBandToVRT(const std::string &fname, GDALDataType dtype);
      //void close() { GDALClose( _dataset ); }  // todo: fix segfault conflict with destructor

      //Pixel read/write with buffer passed by reference, optional band index
      /** Get/Set single value for given band */
      template<typename T> void getSetValue(T& buffer, size_t xidz, size_t yidx, size_t band, GDALRWFlag);
      template<typename T> void getValue(T& buffer, size_t xidx, size_t yidx, size_t band = 1);
      template<typename T> void setValue(T& buffer, size_t xidx, size_t yidx, size_t band = 1);

      // Line read/write with raw pointer and width or STL container, optional band index
      /** Get/Set line in a band from raw pointer */
      template<typename T> void getSetLine(T* buffer, size_t yidx, size_t iowidth, size_t band, GDALRWFlag iodir);
      /** Read one line of data from given band to buffer, vector, or valarray */
      template<typename T> void getLine(T* buffer,             size_t yidx, size_t iowidth, size_t band = 1);
      template<typename T> void getLine(std::vector<T>& vec,   size_t yidx, size_t band = 1);
      template<typename T> void getLine(std::valarray<T>& arr, size_t yidx, size_t band = 1);
      /** Write one line of data from buffer, vector, or valarray to given band*/
      template<typename T> void setLine(T* buffer,             size_t yidx, size_t iowidth, size_t band = 1);
      template<typename T> void setLine(std::vector<T>& vec,   size_t yidx, size_t band = 1);
      template<typename T> void setLine(std::valarray<T>& arr, size_t yidx, size_t band = 1);

      // 2D block read/write, generic container w/ width or STL container, optional band index
      /** Get/Set block in band from raw pointer */
      template<typename T> void getSetBlock(T* buffer,          size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band, GDALRWFlag iodir);
      /** Read block of data from given band to buffer, vector, or valarray */
      template<typename T> void getBlock(T* buffer,             size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band = 1);
      template<typename T> void getBlock(std::vector<T>& vec,   size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band = 1);
      template<typename T> void getBlock(std::valarray<T>& arr, size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band = 1);
      /** Write block of data to given band from buffer, vector, or valarray */
      template<typename T> void setBlock(T* buffer,             size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band = 1);
      template<typename T> void setBlock(std::vector<T>& vec,   size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band = 1);
      template<typename T> void setBlock(std::valarray<T>& arr, size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band = 1);

      //2D block read/write for Matrix<T>, optional band index
      /** Read/write block of data from given band to/from Matrix<T> */
      template<typename T> void getBlock(isce::core::Matrix<T>& mat, size_t xidx, size_t yidx, size_t band = 1);
      template<typename T> void setBlock(isce::core::Matrix<T>& mat, size_t xidx, size_t yidx, size_t band = 1);
      //2D block read/write for Matrix<T>, optional band index
      /** Read/Write block of data from given band to/from Matrix<T>::view_type */
      template<typename T> void getSetBlock(pyre::grid::View<T>& view, size_t xidx, size_t yidx, size_t band, GDALRWFlag iodir);
      template<typename T> void    getBlock(pyre::grid::View<T>& view, size_t xidx, size_t yidx, size_t band = 1);
      template<typename T> void    setBlock(pyre::grid::View<T>& view, size_t xidx, size_t yidx, size_t band = 1);

      //Functions to deal with projections and geotransform information
      /** Return EPSG code corresponding to raster*/
      int getEPSG();
      /** Set EPSG code*/
      int setEPSG(int code);
      /** Set Raster GeoTransform from buffer, vector, or valarray */
      inline void setGeoTransform(double *arr);
      inline void setGeoTransform(std::vector<double>&);
      inline void setGeoTransform(std::valarray<double>&);
      /** Copy Raster GeoTransform into a buffer, vector, or valarray */
      inline void getGeoTransform(double *) const;
      inline void getGeoTransform(std::vector<double>&) const;
      inline void getGeoTransform(std::valarray<double>&) const;
      //Read only functions for specific elements of GeoTransform
      /** Return Western Limit of Raster*/
      inline double x0() const;
      /** Return Northern Limit of Raster*/
      inline double y0() const;
      /** Return EW pixel spacing of Raster*/
      inline double dx() const;
      /** Return NS pixel spacing of Raster*/
      inline double dy() const;

private:
    GDALDataset * _dataset;

};

#define ISCE_IO_RASTER_ICC
#include "Raster.icc"
#undef ISCE_IO_RASTER_ICC

#endif
