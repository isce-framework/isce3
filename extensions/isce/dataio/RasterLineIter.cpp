//
// Author: Joshua Cohen
// Copyright 2017
//

#include "gdal_priv.h"
#include "RasterLineIter.h"


template<typename T>
const T* RasterLineIter<T>::operator*() {
    /*
     * Rvalue indirection operator. For now to save re-access time, this reads the current line into
     * the internal buffer and returns the pointer to the buffer. Since this is the rvalue
     * indirection, we don't have to worry about the user assuming this sets the line.
     */
    // We're using getSetLine because getLine currently uses the vector<> container (will probably
    // change). Check first to see if _linebuf already contains the line we want.
    if (_lineidx != _linebufidx) {
        // Read the line pointed to by the iterator
        _raster->getSetLine(_linebuf, _lineidx, _linewidth, _rasterband, false);
        // Update the iterator info to be aware that we've read line _lineidx to _linebuf
        _linebufidx = _lineidx;
    }
    // Send back the pointer to the buffer. Since we're qualifying this as the const form we
    // don't have to worry about the user modifying this buffer memory.
    return _linebuf;
}

template<typename T>
const T* getLineSequential() {
    /*
     * Essentially the same as dereferencing the iterator. Leaving this in here for legacy
     * semantics, but it's a pass-through for dereferencing.
     */
    // Read the line if needed. out will be the same pointer as _linebuf
    const T *out = operator*();
    // Increment the iterator
    operator++();
    return out;
}

template<typename T>
void setLineSequential(const T *inbuf) {
    /*
     * Write the next line to the Raster file. We don't want to copy the data entirely, so ignore
     * _linebuf and write straight from the input.
     *
     * DEV NOTE: Two things. One, we can only assume that inbuf is long enough to fill a line in
     * the image (since we can't get size from an arbitrary container pointer). Two, we may want
     * to create two different Raster iterators, one to serve as input, one to serve as output.
     * This might allow us to comply fully with "output iterator" design, which requires the
     * indirection operator to return as an lvalue reference (we can't really do that with this
     * iterator given how the rvalue reference indirection returns).
     */
    // Write the line to file
    _raster->getSetLine(inbuf, _lineidx, _linewidth, _rasterband, true);
    // Increment the iterator
    operator++();
}

