//
// Author: Joshua Cohen
// Copyright 2017
//

#include <array>
#include <iostream>
#include <vector>
#include "RasterLineIter.h"
using std::array;
using std::cout;
using std::endl;
using std::vector;
using isce::dataio::RasterLineIter;

template<typename T>
void RasterLineIter::getNext(T *buffer, size_t width, size_t band) {
    /*
     *  Gets the current line pointed to by _lineidx and increments the iterator. Since this uses
     *  an arbitrary memory buffer, we need to know how wide the buffer is being passed in (using
     *  width parameter). Points to a specific band in the image.
     */
    // Check to see if we have the off-the-end iterator
    if (!atEOF()) {
        // Read line pointed to by _lineidx
        _raster->getLine(buffer, _lineidx, width, band);
        operator++;
    } else {
        cout << "In RasterLineIter::getNext() - Iterator has reached EOF." << endl;
    }
}

template<typename T>
void RasterLineIter::getNext(T *buffer, size_t width) {
    /*
     *  Gets the current line pointed to by _lineidx and increments the iterator. Needs to know the
     *  width of the buffer since it's a raw pointer. Points to the default band.
     */
    // Check to see if we have the off-the-end iterator
    getNext(buffer, width, 1);
}

template<typename T>
void RasterLineIter::getNext(array<T> &buffer, size_t band) {
    /*
     *  Gets the current line pointed to by _lineidx and increments the iterator. The width of the
     *  container is inferred by the Raster::getLine() function. Points to a specific image band.
     */
    getNext(buffer.data(), buffer.size(), band);
}

template<typename T>
void RasterLineIter::getNext(array<T> &buffer) {
    /*
     *  Gets the current line pointed to by _lineidx and increments the iterator. Points to the
     *  default image band.
     */
    getNext(buffer.data(), buffer.size());
}

template<typename T>
void RasterLineIter::getNext(vector<T> &buffer, size_t band) {
    /*
     *  Gets the current line pointed to by _lineidx and increments the iterator. The width of the
     *  container is inferred by the Raster::getLine() function. Points to a specific image band.
     */
    getNext(buffer.data(), buffer.size(), band);
}

template<typename T>
void RasterLineIter::getNext(vector<T> &buffer) {
    /*
     *  Gets the current line pointed to by _lineidx and increments the iterator. Points to the
     *  default image band.
     */
    getNext(buffer.data(), buffer.size());
}

template<typename T>
void RasterLineIter::setNext(T *buffer, size_t width, size_t band) {
    /*
     *  Sets the current line pointed to by _lineidx and increments the iterator. Points to a 
     *  specific image band.
     */
    if (!atEOF()) {
        _raster->setLine(buffer, _lineidx, width, band);
        operator++;
    } else {
        cout << "In RasterLineIter::setNext() - Iterator has reached EOF." << endl;
    }
}

template<typename T>
void RasterLineIter::setNext(T *buffer, size_t width) {
    /*
     *  Sets the current line pointed to by _lineidx and increments the iterator. Points to the
     *  default image band.
     */
    setNext(buffer, width, 1);
}

template<typename T>
void RasterLineIter::setNext(array<T> &buffer, size_t band) {
    /*
     *  Sets the current line pointed to by _lineidx and increments the iterator. Points to a
     *  specific image band. As with other I/O methods using STL containers as a buffer, the width
     *  is derived from the container itself and doesn't need to be explicitly passed in.
     */
    setNext(buffer.data(), buffer.size(), band);
}

template<typename T>
void RasterLineIter::setNext(array<T> &buffer) {
    /*
     *  Sets the current line pointed to by _lineidx and increments the iterator. Points to the
     *  default image band.
     */
    setNext(buffer.data(), buffer.size());
}

template<typename T>
void RasterLineIter::setNext(vector<T> &buffer, size_t band) {
    /*
     *  Sets the current line pointed to by _lineidx and increments the iterator. Points to a
     *  specific image band.
     */
    setNext(buffer.data(), buffer.size(), band);
}

template<typename T>
void RasterLineIter::setNext(vector<T> &buffer) {
    /*
     *  Sets the current line pointed to by _lineidx and increments the iterator. Points to the
     *  default image band.
     */
    setNext(buffer.data(), buffer.size());
}

