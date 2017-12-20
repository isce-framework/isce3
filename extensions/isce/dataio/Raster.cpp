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
using std::length_error;
using std::min;
using std::runtime_error;
using std::string;
using std::vector;
using isce::dataio::Raster;


// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
//                                          OBJECT MANAGEMENT
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
Raster::Raster() : _dataset(nullptr), _readonly(true) {
    /*
     *  Empty constructor also handles initializing the GDAL Drivers (if needed).
     */
    // Check if we've already registered the drivers by trying to load the VRT driver (the choice of
    // driver is arbitrary)
    if (GetGDALDriverManager()->GetDriverByName("VRT") == nullptr) {
        GDALAllRegister();
    }
}

Raster::Raster(const string &fname, bool readOnly=true) : _readonly(readOnly) {
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

Raster::Raster(const Raster &rast) : _dataset(rast._dataset), _readonly(rast._readonly) {
    /*
     *  Slightly enhanced copy constructor since we can't simply weak-copy the GDALDataset* pointer.
     *  Increment the reference to the GDALDataset's internal reference counter after weak-copying
     *  the pointer.
     */
    if (_dataset) _dataset->Reference();
}

Raster::~Raster() {
    /*
     *  To account for the fact that multiple Raster objects might reference the same dataset, and
     *  to avoid duplicating the resources GDAL allocates under the hood, we work with GDAL's 
     *  reference/release system to handle GDALDataset* management. If you call release and the
     *  dataset's reference counter is 1, it automatically deletes the dataset.
     */
    if (_dataset) _dataset->Release();
}

void Raster::loadFrom(const string &fname, bool readOnly=true) {
    /*
     *  Provides the same functionality as Raster::Raster(string&,bool), but assumes that the
     *  GDALDrivers have already been initialized (since this can't be called as a static method).
     *  Will also try to delete any existing dataset that has been previously loaded (releasing
     *  stored resources).
     */
    // Dereference the currently-loaded GDALDataset (safer way to 'delete' since the loaded
    // GDALDataset might have come from copy-construction
    _dataset->Release();
    if (readOnly) _dataset = static_cast<GDALDataset*>(GDALOpen(fname.data(), GA_ReadOnly));
    else _dataset = static_cast<GDALDataset*>(GDALOpen(fname.data(), GA_Update));
    _readonly = readOnly;
    // Note that in most cases if GDALOpen fails it will throw a CPL_ERROR anyways
    if (_dataset == nullptr) {
        string errstr = "In Raster::loadFrom() - Cannot open file '";
        errstr += fname;
        errstr += "'.";
        throw runtime_error(errstr);
    }
}


// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
//                                          PIXEL OPERATIONS
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
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
     *  Gets a single pixel at a given location in the image from a given raster band.
     */
    // False as the final value indicates we want to get the value (true means set)
    getSetValue(buffer, xidx, yidx, band, false);
}

template<typename T>
void Raster::getValue(T &buffer, size_t xidx, size_t yidx) {
    /*
     *  Gets a single pixel at a given location in the image with a default raster band. Note this
     *  is only necessary because of a legacy requirement on default parameters and template methods.
     */
    getValue(buffer, xidx, yidx, 1, false);
}

template<typename T>
void Raster::setValue(T &buffer, size_t xidx, size_t yidx, size_t band) {
    /*
     *  Sets a single pixel at a given location in the image in a given raster band.
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
     *  Sets a single pixel at a given location in the image in the default raster band.
     */
    setValue(buffer, xidx, yidx, 1, false);
}


// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
//                                          LINE OPERATIONS
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template<typename T>
void Raster::getSetLine(T *buffer, size_t line, size_t iowidth, size_t band, bool set) {
    /*
     *  As with getSetValue, this serves as the single entry point for reading/writing a line of
     *  data to/from the image. Note that this function differs slightly from the caller signature
     *  because we want this to be able to accept any line of data in contiguous memory (i.e. not
     *  restricting it to just accepting std::vectors). The added cost is the callers need to pass
     *  in the width of the data to be written/read, since we cannot find that information from the
     *  buffer pointer.
     */
    // Test if the buffer datatype is mappable to a GDALDataType
    if (_gdts.count(typeid(T))) {
        // Check if we have an image loaded
        if (_dataset) {
            // Check bounds
            if ((line < getLength()) && (band <= getNumBands())) {
                // Determine I/O direction
                GDALDataType iodir = set ? GF_Write : GF_Read;
                // Determine number of elements to read (smaller of requested number of elements
                // and number of elements in the image line)
                size_t rdwidth = min(iowidth, getWidth());
                auto _ = _dataset->GetRasterBand(band)->RasterIO(iodir, 0, line, rdwidth, 1, buffer,
                                                                 rdwidth, 1, _gdts.at(typeid(T)), 0,
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
void Raster::getLine(T *buffer, size_t line, size_t iowidth, size_t band) {
    /*
     *  Gets a single line at a given location in the image from a given raster band. Also accounts
     *  for iowidth, which is the size of the buffer. This needs to be passed in explicitly since it
     *  can't be derived from the buffer pointer itself.
     */
    // False as the last value indicates we want to read from the image (true indicates write)
    getSetLine(buffer, line, iowidth, band, false);
}

template<typename T>
void Raster::getLine(T *buffer, size_t line, size_t iowidth) {
    /*
     *  Gets a single line at a given location in the image from the default raster band. Also
     *  accounts for iowidth, which is the size of the buffer.
     */
    getLine(buffer, line, iowidth, 1);
}

template<typename T>
void Raster::getLine(vector<T> &buffer, size_t line, size_t band) {
    /*
     *  Gets a single line at a given location in the image from a given raster band. As above, we
     *  derive iowidth from the actual STL container.
     */
    getLine(buffer.data(), line, buffer.size(), band);
}

template<typename T>
void Raster::getLine(vector<T> &buffer, size_t line) {
    /*
     *  Gets a single line at a given location in the image from the default raster band. As above,
     *  we derive iowidth from the actual STL container.
     */
    getLine(buffer.data(), line, buffer.size(), 1);
}

template<typename T>
void Raster::setLine(T* buffer, size_t line, size_t iowidth, size_t band) {
    /*
     *  Sets a single line at a given location in the image in a given raster band. Also accounts
     *  for iowidth, which is the size of the buffer. This needs to be passed in explicitly since it
     *  can't be derived from the buffer pointer itself.
     */
    // Check if we are even allowed to write to the image (i.e. check if we opened in read-only)
    if (!_readonly) {
        // True as the final value indicates we want to write to the image (false means read)
        getSetLine(buffer, line, iowidth, band, true);
    } else {
        cout << "In Raster::setLine() - Image was opened in read-only mode and cannot be " <<
                "written to." << endl;
    }
}

template<typename T>
void Raster::setLine(T* buffer, size_t line, size_t iowidth) {
    /*
     *  Sets a single line at a given location in the image in the default raster band. Also
     *  accounts for iowidth, which is the size of the buffer.
     */
    setLine(buffer, line, iowidth, 1);
}

template<typename T>
void Raster::setLine(vector<T> &buffer, size_t line, size_t band) {
    /*
     *  Sets a single line at a given location in the image in a given raster band. As above, we
     *  derive iowidth from the actual STL container.
     */
    setLine(buffer.data(), line, buffer.size(), band);
}

template<typename T>
void Raster::setLine(vector<T> &buffer, size_t line) {
    /*
     *  Sets a single line at a given location in the image in the default raster band. As above,
     *  we derive iowidth from the actual STL container.
     */
    setLine(buffer.data(), line, buffer.size(), 1);
}


// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
//                                      BLOCK OPERATIONS
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

template<typename T>
void Raster::getSetBlock(T *buffer, size_t xidx, size_t yidx, size_t iolength, size_t iowidth,
                         size_t band, bool set) {
    /*
     *  As with getSetValue/getSetLine, this serves as a single entry point for reading/writing a
     *  block of data to/from the image. The key thing to note about this function is that due to
     *  the way GDAL interfaces with files, the buffer needs to be a 1D-contiguous block of memory
     *  for efficient read/write block operations. This means buffer must be 1D and linearly
     *  contiguous. We also need to know (in addition to the positional offset in the image), both
     *  the length and width of the buffer.
     *
     *  The only concern at the moment is I'm unsure how exactly this maps to RasterIO underneath.
     *  The implementation below should be the correct API call, but this needs to be tested.
     */
    // Check to see if the buffer datatype is mappable to a GDALDataType
    if (_gdts.count(typeid(T))) {
        // Check if we have an image loaded
        if (_dataset) {
            // Check bounds
            if (((xidx + iolength) < getLength()) && ((yidx + iowidth) < getWidth()) &&
                (band <= getNumBands())) {
                // Determine I/O direction based on whether the "set" flag was set (true ==
                // GF_Write, false == GF_Read)
                GDALDataType iodir = set ? GF_Write : GF_Read;
                // As with getSetValue/getSetLine, use the typeid magic to translate datatypes on
                // the fly with RTTI.
                auto _ = _dataset->GetRasterBand(band)->RasterIO(iodir, xidx, yidx, iowidth,
                                                                 iolength, buffer, iowidth,
                                                                 iolength, _gdts.at(typeid(T)),
                                                                 0, 0);
            } else {
                throw domain_error("In Raster::get/setBlock() - 2D/band index is out-of-bounds.");
            }
        } else {
            cout << "In Raster::get/setBlock() - No dataset loaded." << endl;
        }
    } else {
        cout << "In Raster::get/setBlock() - Buffer datatype (type " << typeid(T).name() << ") " <<
                "is not mappable to a GDALDataType." << endl;
    }
}

template<typename T>
void Raster::getBlock(T *buffer, size_t xidx, size_t yidx, size_t iolength, size_t iowidth, 
                      size_t band) {
    /*
     *  Gets a block at a given location in the image from a given raster band. Needs to account for
     *  both block dimensions since we're passing in a 1D pointer.
     */
    // False as the last parameter indicates we want to read from the image (true == write)
    getSetBlock(buffer, xidx, yidx, iolength, iowidth, band, false);
}

template<typename T>
void Raster::getBlock(T *buffer, size_t xidx, size_t yidx, size_t iolength, size_t iowidth) {
    /*
     *  Gets a block at a given location in the image from the default raster band.
     */
    getSetBlock(buffer, xidx, yidx, iolength, iowidth, 1, false);
}

template<typename T>
void Raster::getBlock(vector<T> &buffer, size_t xidx, size_t yidx, size_t iolength, size_t iowidth,
                      size_t band) {
    /*
     *  Gets a block at a given location in the image from a given raster band. Even though the
     *  caller passes in an STL container, we need to know the 2D layout of the data the caller
     *  wants (inferred from iolength/iowidth). We can provide an extra layer of security though
     *  and make sure the container can hold the requested data (we can only check the size of the
     *  container against the number of requested elements). Note that if there's a size mismatch
     *  (i.e. if iolength*iowidth < buffer.size()), that will interfere with the layout of the data
     *  in the buffer, which may not be transparent to the caller if it was unintentional. The best
     *  we can do in this case is report the size mismatch to the caller.
     */
    // Check for valid buffer sizing
    if ((iolength * iowidth) <= buffer.size()) {
        getBlock(buffer.data(), xidx, yidx, iolength, iowidth, band);
        // Check if there's a size mismatch
        if ((iolength * iowidth) < buffer.size()) {
            cout << "In Raster::getBlock() - Requested fewer elements than the buffer can fit. " <<
                    "Internal data layout in the buffer may be different than expected." << endl;
        }
    } else {
        throw length_error("In Raster::getBlock() - Requested more elements than the buffer size.");
    }
}

template<typename T>
void Raster::getBlock(vector<T> &buffer, size_t xidx, size_t yidx, size_t iolength, 
                      size_t iowidth) {
    /*
     *  Gets a block at a given location in the image from the default raster band.
     */
    getBlock(buffer, xidx, yidx, iolength, iowidth, 1);
}

template<typename T>
void Raster::getBlock(vector<vector<T>> &buffer, size_t xidx, size_t yidx, size_t band) {
    /*
     *  Gets a block at a given location in the image from a given raster band. In this case, since
     *  the caller is passing an implicitly-size 2D STL container, we can check sizing appropriately
     *  before trying to read data (i.e. make sure the width/height of the STL container is valid).
     *
     *  NOTE: THIS IMPLEMENTATION ACTUALLY CALLS ITERATIONS OF BLOCK READER AND MAY BE SIGNIFICANTLY
     *  SLOWER THAN THE OTHER BLOCK READERS. THIS METHOD IS ONLY IMPLEMENTED FOR CALLER CONVENIENCE
     *  AND IS NOT RECOMMENDED FOR FAST I/O NEEDS.
     */
    // Check boundaries of attempted block read (this is mostly to make sure size_t arithmetic
    // doesn't have unexpected results)
    if ((xidx < getLength()) && (yidx < getWidth()) && (band <= getNumBands())) {
        // Number of lines to read is minimum of number of lines available (based on xidx) and the
        // length of the container
        size_t iolength = min(getLength()-xidx, buffer.size());
        for (size_t line=0; line<iolength; line++) {
            // Call getBlock() on a 1D vector (basically getLine with an offset)
            getBlock(buffer[line], xidx+line, yidx, 1, buffer[line].size(), band);
        }
    } else {
        throw domain_error("In Raster::getBlock() - 2D/band index is out-of-bounds.");
    }
}

template<typename T>
void Raster::getBlock(vector<vector<T>> &buffer, size_t xidx, size_t yidx) {
    /*
     *  Gets a block at a given location in the image from the default raster band. As above, sizing
     *  is determined by the STL container methods.
     */
    getBlock(buffer, xidx, yidx, 1);
}

template<typename T>
void Raster::setBlock(T *buffer, size_t xidx, size_t yidx, size_t iolength, size_t iowidth,
                      size_t band) {
    /*
     *  Sets a block at a given location in the image to a given raster band. Needs to account for
     *  both block dimensions since we're passing in a 1D pointer.
     */
    // True as the last parameter indicates we want to write to the image (false == read)
    getSetBlock(buffer, xidx, yidx, iolength, iowidth, band, true);
}

template<typename T>
void Raster::setBlock(T *buffer, size_t xidx, size_t yidx, size_t iolength, size_t iowidth) {
    /*
     *  Sets a block at a given location in the image to the default raster band.
     */
    getSetBlock(buffer, xidx, yidx, iolength, iowidth, 1, true);
}

template<typename T>
void Raster::setBlock(vector<T> &buffer, size_t xidx, size_t yidx, size_t iolength, size_t iowidth,
                      size_t band) {
    /*
     *  Sets a block at a given location in the image to a given raster band. As with getBlock(),
     *  we need to know the 2D layout of the data the caller wants, so we check for size mismatch.
     *  In the case of writing the data, if you try to write fewer elements than the input buffer
     *  contains, you may be writing the wrong shape/size of the data to the file. Therefore we try
     *  to at least alert the caller that this might be the case (in case it was unintentional).
     */
    // Check for valid buffer sizing (otherwise we'll write data from outside the container to the
    // file; BIG security hole if left unchecked)
    if ((iolength * iowidth) <= buffer.size()) {
        setBlock(buffer.data(), xidx, yidx, iolength, iowidth, band);
        // Check for size mismatch
        if ((iolength * iowidth) < buffer.size()) {
            cout << "In Raster::setBlock() - Writing fewer elements than the buffer is sized " <<
                    "for. Data layout in the file may be different than expected." << endl;
        }
    } else {
        throw length_error("In Raster::setBlock() - Setting more elements than the buffer size.");
    }
}

template<typename T>
void Raster::setBlock(vector<T> &buffer, size_t xidx, size_t yidx, size_t iolength,
                      size_t iowidth) {
    /*
     *  Sets a block at a given location in the image to the default raster band.
     */
    setBlock(buffer, xidx, yidx, iolength, iowidth, 1);
}

template<typename T>
void Raster::setBlock(vector<vector<T>> &buffer, size_t xidx, size_t yidx, size_t band) {
    /*
     *  Sets a block at a given location in the image to a given raster band. In this case, since
     *  the caller is passing an implicitly-sized 2D STL container, we can check sizing
     *  appropriately before trying to read data (i.e. make sure the width/height of the STL
     *  container is valid).
     *
     *  NOTE: THIS IMPLEMENTATION ACTUALLY CALLS ITERATIONS OF BLOCK WRITER AND MAY BE SIGNIFICANTLY
     *  SLOWER THAN THE OTHER BLOCK WRITERS. THIS METHOD IS ONLY IMPLEMENTED FOR CALLER CONVENIENCE
     *  AND IS NOT RECOMMENDED FOR FAST I/O NEEDS.
     */
    // Check boundaries of attempted block write
    if ((xidx < getLength()) && (yidx < getWidth()) && (band <= getNumBands())) {
        // Number of lines to read is min number of lines available (based on xidx) and the length
        // of the container
        size_t iolength = min(getLength()-xidx, buffer.size());
        for (size_t line=0; line<iolength; line++) {
            setBlock(buffer[line], xidx+line, yidx, 1, buffer[line].size(), band);
        }
    } else {
        throw domain_error("In Raster::setBlock() - 2D/band index is out-of-bounds.");
    }
}

template<typename T>
void Raster::setBlock(vector<vector<T>> &buffer, size_t xidx, size_t yidx) {
    /*
     *  Sets a block at a given location in the image to the default raster band. As above, sizing
     *  is determined by the STL container methods.
     */
    setBlock(buffer, xidx, yidx, 1);
}

