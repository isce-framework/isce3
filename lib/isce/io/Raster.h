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
   
    /** Data structure meant to handle Raster I/O operations.
     *
     * This is currently a thin wrapper over GDAL's Dataset class with some simpler
     * interfaces for I/O. ISCE is expected to only support North-up and West-left 
     * oriented rasters. */
    class Raster {
      
    public:

      Raster() {_dataset = nullptr;};

      /** Constructor to open existing file in ReadOnly mode*/
      Raster(const std::string&);

      /** Constructor to open an existing file with specified Access mode*/
      Raster(const std::string& fname, GDALAccess access);

      /** Constructor to create a dataset*/
      Raster(const std::string& fname, size_t width, size_t length, size_t numBands, GDALDataType dtype, const std::string& driverName);

      /** Constructor to create a dataset with defaultGDALDriver*/
      Raster(const std::string& fname, size_t width, size_t length, size_t numBands, GDALDataType dtype);

      /** Constructor to create a dataset with defaultGDALDriver and defaultGDALDataType*/
      Raster(const std::string& fname, size_t width, size_t length, size_t numBands);

      /** Constructor to create a 1 band dataset with default Driver*/
      Raster(const std::string& fname, size_t width, size_t length, GDALDataType dtype);

      /** Constructor to create a 1 band dataset with default Driver and data type*/
      Raster(const std::string& fname, size_t width, size_t length);

      /** Constructor for a 1 band dataset from isce::core::Matrix<T> */
      template<typename T> Raster(isce::core::Matrix<T> &matrix);

      // Constructor for a 1 band dataset from isce::core::Matrix<T>::view_type 
      template<typename T> Raster(pyre::grid::View<T> &view);

      /** Create new raster object like another */
      Raster(const std::string& fname, const Raster& rast);
      template<typename T> Raster(const std::string&, const std::vector<T>&,   size_t);
      template<typename T> Raster(const std::string&, const std::valarray<T>&, size_t);

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
      /** Get/Set single value */  
      template<typename T> void getSetValue(T& buffer, size_t xidz, size_t yidx, size_t band, GDALRWFlag);
      /** Get single value for given band*/
      template<typename T> void getValue(T& buffer, size_t xidx, size_t yidx, size_t band);
      /**Get single value from band 1*/
      template<typename T> void getValue(T& buffer, size_t xidx, size_t yidx);
      /** Set single value for given band*/
      template<typename T> void setValue(T& buffer, size_t xidx, size_t yidx, size_t band);
      /** Set single value in band 1*/
      template<typename T> void setValue(T& buffer, size_t xidx, size_t yidx);

     
      // Line read/write with raw pointer and width or STL container, optional band index
      /** Get/Set line in a band from raw pointer */
      template<typename T> void getSetLine(T* buffer, size_t yidx, size_t iowidth, size_t band, GDALRWFlag iodir);
      /** Read one line of data from given band to buffer */ 
      template<typename T> void getLine(T* buffer, size_t yidx, size_t iowidth, size_t band);
      /** Read one line of data from band 1 to buffer */
      template<typename T> void getLine(T* buffer, size_t yidx, size_t iowidth);
      /** Read one line of data from given band to std::vector*/
      template<typename T> void getLine(std::vector<T>& vec, size_t yidx, size_t band);
      /** Read one line of data from band 1 to std::vector*/
      template<typename T> void getLine(std::vector<T>& vec, size_t yidx);
      /** Read one line of data from given band to std::valarray*/
      template<typename T> void getLine(std::valarray<T>& arr, size_t yidx, size_t band);
      /** Read one line of data from band 1 to std::valarray*/
      template<typename T> void getLine(std::valarray<T>& arr, size_t yidx);
      /** Write one line of data from buffer to given band*/ 
      template<typename T> void setLine(T* buffer, size_t yidx, size_t iowidth, size_t band);
      /** Write one line of data from buffer to band 1*/
      template<typename T> void setLine(T* buffer, size_t yidx, size_t iowidth);
      /** Write one line of data from std::vector to given band */
      template<typename T> void setLine(std::vector<T>& vec, size_t idx, size_t band);
      /** Write one line of data from std::vector to band 1*/
      template<typename T> void setLine(std::vector<T>& vec, size_t idx);
      /** Write one line of data from std::valarray to given band*/
      template<typename T> void setLine(std::valarray<T>& arr, size_t yidx, size_t band);
      /** Write one line of data from std::valarray to band 1*/
      template<typename T> void setLine(std::valarray<T>& arr, size_t yidx);

      
      // 2D block read/write, generic container w/ width or STL container, optional band index
      /** Get/Set block in band from raw pointer */
      template<typename T> void getSetBlock(T* buffer, size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band, GDALRWFlag iodir);
      /** Read block of data from given band to buffer*/
      template<typename T> void getBlock(T* buffer, size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band);
      /** Read block of data from band 1 to buffer*/
      template<typename T> void getBlock(T* buffer, size_t xidx, size_t yidx, size_t iowidth, size_t iolength);
      /** Read block of data from given band to std::vector*/
      template<typename T> void getBlock(std::vector<T>& vec, size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band);
      /** Read block of data from band 1 to std::vector*/
      template<typename T> void getBlock(std::vector<T>& vec, size_t xidx, size_t yidx, size_t iowidth, size_t iolength);
      /** Read block of data from given band to std::valarray*/
      template<typename T> void getBlock(std::valarray<T>& arr, size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band);
      /** Read block of data from band 1 to std::valarray */
      template<typename T> void getBlock(std::valarray<T>& arr, size_t xidx, size_t yidx, size_t iowidth, size_t iolength);
      /** Write block of data to given band from buffer*/
      template<typename T> void setBlock(T* buffer, size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band);
      /** Write block of data to band 1 from buffer*/
      template<typename T> void setBlock(T* buffer, size_t xidx, size_t yidx, size_t iowidth, size_t iolength);
      /** Write block of data to given band from std::vector*/
      template<typename T> void setBlock(std::vector<T>& vec, size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band);
      /** Write block of data to band 1 from std::vector*/
      template<typename T> void setBlock(std::vector<T>& vec, size_t xidx, size_t yidx, size_t iowidth, size_t iolength);
      /** Write block of data to given band from std::valarray*/
      template<typename T> void setBlock(std::valarray<T>& arr, size_t xidx, size_t yidx, size_t iowidth, size_t iolength, size_t band);
      /** Write block of data to band 1 from std::valarray*/
      template<typename T> void setBlock(std::valarray<T>& arr, size_t xidx, size_t yidx, size_t iowidth, size_t iolength);
    
      //2D block read/write for Matrix<T>, optional band index
      /** Read block of data from given band to Matrix<T> */
      template<typename T> void getBlock(isce::core::Matrix<T>& mat, size_t xidx, size_t yidx, size_t band);

      /** Read block of data from band 1 to Matrix<T> */
      template<typename T> void getBlock(isce::core::Matrix<T>& mat, size_t xidx, size_t yidx);
      
      /** Write block of data to given band from Matrix<T> */
      template<typename T> void setBlock(isce::core::Matrix<T>& mat, size_t xidx, size_t yidx, size_t band);

      /** Write block of data to band 1 from Matrix<T> */
      template<typename T> void setBlock(isce::core::Matrix<T>& mat, size_t xidx, size_t yidx); 

      //2D block read/write for Matrix<T>, optional band index
      /** Read/Write block of data from given band to/from Matrix<T>::view_type */
      template<typename T> void getSetBlock(pyre::grid::View<T>& view, size_t xidx, size_t yidx, size_t band, GDALRWFlag iodir);

      /** Read block of data from given band to Matrix<T>::view_type */
      template<typename T> void getBlock(pyre::grid::View<T>& view, size_t xidx, size_t yidx, size_t band);

      /** Read block of data from band 1 to Matrix<T>::view_type */
      template<typename T> void getBlock(pyre::grid::View<T>& view, size_t xidx, size_t yidx);

      /** Write block of data to given band from Matrix<T>::view_type */
      template<typename T> void setBlock(pyre::grid::View<T>& view, size_t xidx, size_t yidx, size_t band);

      /** Write block of data to band 1 from Matrix<T>::view_type */
      template<typename T> void setBlock(pyre::grid::View<T>& view, size_t xidx, size_t yidx);

      //Functions to deal with projections and geotransform information
      /** Return EPSG code corresponding to raster*/
      int getEPSG();
      /** Set EPSG code*/
      int setEPSG(int code);
      /** Set Raster GeoTransform from buffer*/
      inline void setGeoTransform(double *arr);
      /** Set Raster GeoTransform from std::vector*/
      inline void setGeoTransform(std::vector<double>&);
      /** Set Raster GeoTransform from std::valarray*/
      inline void setGeoTransform(std::valarray<double>&);
      /** Copy Raster GeoTransform into a buffer */
      inline void getGeoTransform(double *) const;
      /** Copy Raster GeoTransform into std::vector*/
      inline void getGeoTransform(std::vector<double>&) const;
      /** Copy Raster GeoTransform into std::valarray*/
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
  }
}

#define ISCE_IO_RASTER_ICC
#include "Raster.icc"
#undef ISCE_IO_RASTER_ICC

#endif
