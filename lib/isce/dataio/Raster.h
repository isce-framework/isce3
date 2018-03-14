//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Joshua Cohen
// Update: ml
// Copyright 2018
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

namespace isce {
  namespace dataio {

    class Raster {
      // Since we need to pass an output datatype to GDAL's RasterIO using the GDALDataTypes,
      // we can use an unordered_map to map typeids to GDALDataTypes so that we can use RTTI to
      // convert on the fly. Note that this assumes the user's system hasn't overridden the
      // standard float type to mean the same as double (since there's no standard float32_t/
      // float64_t equivalent). Note that due to the way the compiler builds static types, this
      // needs to be declared here but defined as an inline-like implementation below.
      
    public:

      GDALDataset *dataset;
      
      Raster();
      Raster(const std::string&,bool);
      Raster(const Raster&);
      inline Raster& operator=(const Raster&);
      ~Raster();
      
      void loadFrom(const std::string&,bool);
      inline size_t length()   const { return dataset ? dataset->GetRasterYSize() : 0; }
      inline size_t width()    const { return dataset ? dataset->GetRasterXSize() : 0; }
      inline size_t numBands() const { return dataset ? dataset->GetRasterCount() : 0; }
      
      
      /*
       *  Pixel operations
       */
      // Single pixel read/write
      // (buffer, x-index, y-index, band-index, read/write)
      template<typename T> void getSetValue(T&,size_t,size_t,size_t,bool);

      // Single pixel read, optional band index
      // (buffer, x-index, y-index, [band-index])
      template<typename T> void getValue(T&,size_t,size_t,size_t);
      template<typename T> void getValue(T&,size_t,size_t);

      // Single pixel write, optional band index
      // (buffer, x-index, y-index, [band-index])
      template<typename T> void setValue(T&,size_t,size_t,size_t);
      template<typename T> void setValue(T&,size_t,size_t);

      /*
       *  Line operations
       */
      // Single line read/write
      // (buffer, line-index, nelem, band-index, read/write)
      template<typename T> void getSetLine(T*,size_t,size_t,size_t,bool);

      // Single line read, non-specific container w/ container width, optional band index
      // (buffer, line-index, nelem, [band-index])
      template<typename T> void getLine(T*,size_t,size_t,size_t);
      template<typename T> void getLine(T*,size_t,size_t);

      // Single line read, STL containers, optional band index
      // (buffer, line-index, [band-index])
      template<typename T> void getLine(std::vector<T>&,size_t,size_t);
      template<typename T> void getLine(std::vector<T>&,size_t);

      // Single line write, non-specific container w/ container width, optional band index
      // (buffer, line-index, nelem, [band-index])
      template<typename T> void setLine(T*,size_t,size_t,size_t);
      template<typename T> void setLine(T*,size_t,size_t);

      // Single line write, STL containers, optional band index
      // (buffer, line-index, [band-index])
      template<typename T> void setLine(std::vector<T>&,size_t,size_t);
      template<typename T> void setLine(std::vector<T>&,size_t);

      /*
       *  Block operations
       */
      // Single block read/write
      // (buffer, x-index, y-index, nXelem, nYelem, band-index, read/write)
      template<typename T> void getSetBlock(T*,size_t,size_t,size_t,size_t,size_t,bool);

      // Single block read, linear containers w/ x/y sizes, optional band index
      // (buffer, x-index, y-index, nXelem, nYelem, [band-index])
      template<typename T> void getBlock(T*,size_t,size_t,size_t,size_t,size_t);
      template<typename T> void getBlock(T*,size_t,size_t,size_t,size_t);
      template<typename T> void getBlock(std::vector<T>&,size_t,size_t,size_t,size_t,size_t);
      template<typename T> void getBlock(std::vector<T>&,size_t,size_t,size_t,size_t);

      // Single block read, 2D STL containers, optional band index
      // (buffer, x-index, y-index, [band-index])
      template<typename T> void getBlock(std::vector<std::vector<T>>&,size_t,size_t,size_t);
      template<typename T> void getBlock(std::vector<std::vector<T>>&,size_t,size_t);

      // Single block write, linear containers w/ x/y sizes, optional band index
      // (buffer, x-index, y-index, nXelem, nYelem, [band-index])
      template<typename T> void setBlock(T*,size_t,size_t,size_t,size_t,size_t);
      template<typename T> void setBlock(T*,size_t,size_t,size_t,size_t);
      template<typename T> void setBlock(std::vector<T>&,size_t,size_t,size_t,size_t,size_t);
      template<typename T> void setBlock(std::vector<T>&,size_t,size_t,size_t,size_t);

      // Single block write, 2D STL containers, optional band index
      // (buffer, x-index, y-index, [band-index])
      template<typename T> void setBlock(std::vector<std::vector<T>>&,size_t,size_t,size_t);
      template<typename T> void setBlock(std::vector<std::vector<T>>&,size_t,size_t);
      
    private:

      static const std::unordered_map<std::type_index,GDALDataType> _gdts;
      bool _readonly;
      
    };

  }
}

#define ISCE_DATAIO_RASTER_ICC
#include "Raster.icc"
#undef ISCE_DATAIO_RASTER_ICC

#endif
