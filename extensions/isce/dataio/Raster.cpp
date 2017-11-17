//
// Author: Joshua Cohen
// Copyright 2017
//

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "gdal_priv.h"
#include "Raster.h"
using std::cout;
using std::domain_error;
using std::endl;
using std::min;
using std::runtime_error;
using std::string;
using std::vector;
using isce::dataio::Raster;


Raster::Raster() : _dataset(nullptr), _linecount(0), _readonly(true) {
    /*
     *  Empty constructor also handles initializing the GDAL Drivers (if needed).
     */
    // Check if we've already registered the drivers by trying to load the VRT driver (the choice of
    // driver is arbitrary)
    if (GetGDALDriverManager()->GetDriverByName("VRT") == nullptr) {
        GDALAllRegister();
    }
}

Raster::Raster(const string &fname, bool readOnly=true) : _linecount(0), _readonly(readOnly) {
    /*
     *  Construct a Raster object referring to a particular image file, so load the dataset now.
     */
    if (GetGDALDriverManager()->GetDriverByName("VRT") == nullptr) {
        GDALAllRegister();
    }
    if (readOnly) _dataset = static_cast<GDALDataset*>(GDALOpen(fname.data(), GA_ReadOnly));
    else _dataset = static_cast<GDALDataset*>(GDALOpen(fname.data(), GA_Update));
    // Note that in most cases if GDALOpen fails it will throw a CPL_ERROR anyways
    if (_dataset == nullptr) {
        string errstr = "In Raster::Raster() - Cannot open file '";
        errstr += fname;
        errstr += "'.";
        throw runtime_error(errstr);
    }
}

Raster::Raster(const Raster &rast) : _dataset(rast._dataset), _linecount(rast._linecount),
                                     _readonly(rast._readonly) {
    /*
     *  Slightly enhanced copy constructor since we can't simply weak-copy the GDALDataset* pointer.
     *  Increment the reference to the GDALDataset's internal reference counter after weak-copying
     *  the pointer.
     */
    _dataset->Reference();
}

Raster::~Raster() {
    /*
     *  To account for the fact that multiple Raster objects might reference the same dataset, and
     *  to avoid duplicating the resources GDAL allocates under the hood, we work with GDAL's 
     *  reference/release system to handle GDALDataset* management. If you call release and the
     *  dataset's reference counter is 1, it automatically deletes the dataset.
     */
    _dataset->Release();
}

void Raster::loadFrom(const string &fname, bool readOnly=true) {
    /*
     *  Provides the same functionality as Raster::Raster(string&,bool), but assumes that the
     *  GDALDrivers have already been initialized (since this can't be called as a static method).
     *  Will also try to delete any existing dataset that has been previously loaded (releasing
     *  stored resources).
     */
    // Remember that delete's behavior doesn't throw an exception if dataset is NULL, so no need to
    // check in advance
    delete _dataset;
    if (readOnly) _dataset = static_cast<GDALDataset*>(GDALOpen(fname.data(), GA_ReadOnly));
    else _dataset = static_cast<GDALDataset*>(GDALOpen(fname.data(), GA_Update));
    _linecount = 0;
    _readonly = readOnly;
    // Note that in most cases if GDALOpen fails it will throw a CPL_ERROR anyways
    if (_dataset == nullptr) {
        string errstr = "In Raster::loadFrom() - Cannot open file '";
        errstr += fname;
        errstr += "'.";
        throw runtime_error(errstr);
    }
}

template<typename T>
void Raster::getSetValue(T &buffer, size_t xidx, size_t yidx, size_t band, bool set) {
    /*
     *  Because get/set happen with the same GDAL function call (RasterIO), simplify the interface
     *  to flip the GF_Read/GF_Write flag depending on the data direction. Because we can check in
     *  constant time the existence of a key in an unordered_map (i.e. whether the buffer type is
     *  mappable to a GDALDataType), we don't need to forward-declare the template prototypes.
     */
    // Check if the datatype of the buffer is mappable to a GDALDataType. Count returns 1 if the key
    // is in the map, 0 if not (since unordered_map is hash-indexed, the key can only exist once!)
    if (_gdts.count(typeid(T))) {
        // Check if we have an image loaded
        if (_dataset) {
            // Check bounds
            if ((xidx < getLength()) && (yidx < getWidth()) && (band <= getNumBands())) {
                // Determine I/O direction based on whether the "set" flag was set (true ==
                // GF_Write, false == GF_Read)
                GDALDataType iodir = set ? GF_Write : GF_Read;
                // Here there be dragons. This uses typeid magic to index the GDALDataTypes map such
                // that we can translate things like std::complex<float> -> GDT_CFLoat32 on the fly
                // using RTTI! This is by far the cleanest way to accomplish this given that GDAL
                // doesn't come with this ability built in (maybe we suggest it to them...?). Also
                // note we're just discarding the result since it's unused.
                auto _ = _dataset->GetRasterBand(band)->RasterIO(iodir, xidx, yidx, 1, 1, &buffer,
                                                                 1, 1, _gdts.at(typeid(T)), 0, 0);
            } else {
                throw domain_error("In Raster::get/setValue() - 2D/band index is out-of-bounds.");
            }
        } else {
            cout << "In Raster::get/setValue() - No dataset loaded." << endl;
        }
    } else {
        cout << "In Raster::get/setValue() - Buffer datatype (type " << typeid(T).name() << ") " <<
                "is not mappable to a GDALDataType." << endl;
    }
}

template<typename T>
void Raster::getValue(T &buffer, size_t xidx, size_t yidx, size_t band) {
    /*
     *  Grab the value at an arbitrary pixel location. Using templates cleans up any sort of
     *  type conversions we'd need. We don't need to worry about checking for parameters, that's
     *  done by the general getSetValue method.
     */
    // False as the final value indicates we want to get the value (true means set)
    getSetValue(buffer, xidx, yidx, band, false);
}

template<typename T>
void Raster::getValue(T &buffer, size_t xidx, size_t yidx) {
    /*
     *  Function overload to get around the dumb legacy requirement preventing default arguments
     *  in template functions. Basically be able to optionally pass in the Raster band (if you don't
     *  explicitly pass a band, it will default to this function as a pass-through, otherwise it
     *  calls the above getValue with the explicit 'band' parameter).
     */
    getValue(buffer, xidx, yidx, 1, false);
}

template<typename T>
void Raster::setValue(T &buffer, size_t xidx, size_t yidx, size_t band) {
    /*
     *  Set the value at an arbitrary pixel location. Using templates cleans up any sort of
     *  type conversions we'd need. We don't need to worry about checking for parameters, that's
     *  done by the general getSetValue method.
     */
    // Check first if we are even allowed to write to the image (opened in read-only mode or not)
    if (!_readonly) {
        // True as the final value indicates we want to set the value (false means get)
        getSetValue(buffer, xidx, yidx, band, true);
    } else {
        cout << "In Raster::setValue() - Image was opened in read-only mode and cannot be " <<
                "written to." << endl;
    }
}

template<typename T>
void Raster::setValue(T &buffer, size_t xidx, size_t yidx) {
    /*
     *  Overload to treat image band number as a pseudo-default argument. Passing a band number
     *  calls the above setValue, otherwise it calls this as a pass-through.
     */
    setValue(buffer, xidx, yidx, 1, false);
}

template<typename T>
void Raster::getSetLine(T *buffer, size_t line, size_t iowidth, size_t band, bool set) {
    /*
     *  As with getSetValue, this serves as the single entry point for reading/writing a line of
     *  data to/from the image. Note that this function differs slightly from the caller signature
     *  because we want this to be able to accept any line of data in contiguous memory (i.e. not
     *  restricting it to just accepting std::vectors). The added cost is the callers need to pass
     *  in the width of the data to be written/read, since we cannot find that information from the
     *  buffer pointer.
     *  
     *  FUTURE DEV NOTE - Would it make sense to unify getSetValue and getSetLine? Can't do it for
     *                    getSetBlock though (unless we map getSetBlock to be a looped call to
     *                    getSetLine...).
     */
    // Test if the buffer datatype is mappable to a GDALDataType
    if (_gdts.count(typeid(T))) {
        // Check if we have an image loaded
        if (_dataset) {
            // Check bounds
            if ((line < getLength()) && (band <= getNumBands())) {
                // Determine I/O direction
                GDALDataType iodir = set ? GF_Write : GF_Read;
                auto _ = _dataset->GetRasterBand(band)->RasterIO(iodir, 0, line, iowidth, 1, buffer,
                                                                 iowidth, 1, _gdts.at(typeid(T)), 0,
                                                                 0);
            } else {
                throw domain_error("In Raster::get/setLine() - Line/band index is out-of-bounds.");
            }
        } else {
            cout << "In Raster::get/setLine() - No dataset loaded." << endl;
        }
    } else {
        cout << "In Raster::get/setLine() - Buffer datatype (type " << typeid(T).name() << ") " <<
                "is not mappable to a GDALDataType." << endl;
    }
}

template<typename T>
void Raster::getLine(vector<T> &buffer, size_t line, size_t band) {
    /*
     *  Read a given line from the file into buffer. As with get/setValue(), this really is just
     *  a wrapper that indicates to getSetLine which direction the data is moving. This also allows
     *  for the flexibility to write other versions of this method that take other buffer containers
     *  with linearly-contiguous storage.
     */
    // Determine the number of elements to read (the smaller of the width of the buffer and the
    // image width)
    size_t iowidth = min(getWidth(), buffer.size());
    // False as the final value indicates we want to get the line (true means set)
    getSetLine(buffer.data(), line, iowidth, band, false);
}

template<typename T>
void Raster::getLine(vector<T> &buffer, size_t line) {
    /*
     *  Pass-through for getLine with a default band number (see get/setValue for rationale).
     */
    getLine(buffer, line, 1);
}

template<typename T>
void Raster::setLine(vector<T> &buffer, size_t line, size_t band) {
    /*
     *  Write a given line from buffer into the file. As with get/setValue(), this really is just
     *  a wrapper that indicates to getSetLine which direction the data is moving. As with getLine,
     *  this design allows for the flexibility to write other non-std::vector interfaces to
     *  getSetLine (the only requirement is that the buffer container has linearly-contiguous
     *  memory).
     */
    // Check first if we are even allowed to write to the image (opened in read-only mode or not)
    if (!_readonly) {
        // Determine the number of elements to write (the smaller of the width of the buffer and the
        // image width)
        size_t iowidth = min(getWidth(), buffer.size());
        // True as the final value indicates we want to set the line (false means get)
        getSetLine(buffer.data(), line, iowidth, band, true);
    } else {
        cout << "In Raster::setLine() - Image was opened in read-only mode and cannot be " <<
                "written to." << endl;
    }
}

template<typename T>
void Raster::setLine(vector<T> &buffer, size_t line) {
    /*
     *  Pass-through for setLine with a default band number.
     */
    setLine(buffer, line, 1);
}

template<typename T>
void Raster::getLineSequential(vector<T> &buffer, size_t band) {
    /*
     *  Serves as a pass-through for getSetLine, but uses the internal line counter to determine
     *  which line to read (not recommended but conforms with legacy design).
     */
    // Determine number of elements to read
    size_t iowidth = min(getWidth(), buffer.size());
    // Final parameter of false means read the line
    getSetLine(buffer.data(), _linecount++, iowidth, band, false);
}

template<typename T>
void Raster::getLineSequential(vector<T> &buffer) {
    /*
     *  Pass-through for getLineSequential with a default band number.
     */
    getLineSequential(buffer, 1);
}

template<typename T>
void Raster::setLineSequential(vector<T> &buffer, size_t band) {
    /*
     *  Serves as a pass-through for getSetLine, but uses the internal line counter to determine
     *  which line to write (not recommended but conforms with legacy design).
     */
    // Determine number of elements to write
    size_t iowidth = min(getWidth(), buffer.size());
    // Final parameter of true means write the line
    getSetLine(buffer.data(), _linecount++, iowidth, band, true);
}

template<typename T>
void Raster::setLineSequential(vector<T> &buffer) {
    /*
     *  Pass-through for setLineSequential with a default band number.
     */
    setLineSequential(buffer, 1);
}

