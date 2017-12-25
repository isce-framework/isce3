//
// Author: Joshua Cohen
// Copyright 2017
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "RasterBlockIter.h"
using std::cout;
using std::endl;
using std::max;
using std::min;
using std::vector;
using isce::dataio::RasterBlockIter;

RasterBlockIter& RasterBlockIter::operator+=(const size_t rhs) {
    /*
     *  Increment-assignment operator. Handling variable-sized tiling is tricky, but it's just a
     *  matter of sticking to the indexing scheme. To truly treat this class as an iterator, we
     *  establish here the convention that block indices increment first across column-wise, then
     *  row-wise. Example: in a 2x2 matrix, cell [0][0] has index 0, [0][1] has index 1, [1][0] has
     *  index 2, and [1][1] has index 3.
     */
    size_t nBlockPerLine = ceil((1. * _raster.getWidth()) / yskip);
    size_t nBlockPerCol = ceil((1. * _raster.getLength()) / xskip);
    // globalBlock finds the smaller between the linear index plus increment and the off-the-end
    // block (which has an index equal to the total number of blocks)
    size_t globalBlock = min((_xidx * nBlockPerLine) + _yidx + rhs, nBlockPerLine * nBlockPerCol);
    // Now that we know the new "global" index, calculate backwards to the block indices
    _xidx = globalBlock / nBlockPerLine;
    _yidx = globalBlock % nBlockPerLine;
    return *this;
}

RasterBlockIter& RasterBlockIter::operator-=(const size_t rhs) {
    /*
     *  Decrement-assignment operator. See operator+= for indexing specifics and conventions.
     */
    size_t nBlockPerLine = ceil((1. * _raster.getWidth()) / yskip);
    size_t globalBlock = (_xidx * nBlockPerLine) + _yidx;
    // Need to be careful because the indices are unsigned. Unlike operator+=, we need to use
    // slightly different logic to handle 0-boundaries
    _xidx = (rhs <= globalBlock) ? (globalBlock - rhs) / nBlockPerLine : 0;
    _yidx = (rhs <= globalBlock) ? (globalBlock - rhs) % nBlockPerLine : 0;
    return *this;
}

template<typename T>
void RasterBlockIter::getNext(T *buffer, size_t length, size_t width, size_t band) {
    /*
     *  Gets the current block pointed to by _xidx/_yidx and increments the iterator. Since this
     *  uses an arbitrary memory buffer, we need to know the dimensions of the buffer being passed
     *  in (using length/width). Points to a specific band in the image.
     */
    // Check to see if we have the off-the-end iterator
    if (!atEOF()) {
        // Read block pointed to by the indices by using the _xidx/_yidx indices and the xskip/yskip
        // block offsets
        _raster.getBlock(buffer, _xidx*xskip, _yidx*yskip, length, width, band);
        operator++();
    } else {
        cout << "In RasterBlockIter::getNext() - Iterator has reached EOF." << endl;
    }
}

template<typename T>
void RasterBlockIter::getNext(T *buffer, size_t length, size_t width) {
    /*
     *  Gets the current block pointed to by the _xidx/_yidx indices and increments the iterator.
     *  Needs to know the size of the buffer since it's a raw pointer. Points to the default band.
     */
    getNext(buffer, length, width, 1);
}

template<typename T>
void RasterBlockIter::getNext(vector<T> &buffer, size_t length, size_t width, size_t band) {
    /*
     *  Gets the current block pointed to by the _xidx/_yidx indices and increments the iterator.
     *  Needs to know the size of the buffer because it's a linearly-contiguous 1D STL container, so
     *  we can't infer the data layout. Points to a specific image band.
     */
    getNext(buffer.data(), length, width, band);
}

template<typename T>
void RasterBlockIter::getNext(vector<T> &buffer, size_t length, size_t width) {
    /*
     *  Gets the current block pointed to by the _xidx/_yidx indices and increments the iterator. As
     *  above, needs to know the size of the buffer. Points to the default image band.
     */
    getNext(buffer.data(), length, width, 1);
}

template<typename T>
void RasterBlockIter::getNext(vector<vector<T>> &buffer, size_t band) {
    /*
     *  Gets the current block pointed to by the _xidx/_yidx indices and increments the iterator. In
     *  this case of using a 2D STL container buffer, we need to implement the matching getBlock()
     *  that really is a line reader. Points to a specific image band.
     *
     *  NOTE: AS WITH RASTER::GETBLOCK(CONTAINER<CONTAINER<T>>) THIS IMPLEMENTATION ACTUALLY CALLS
     *  ITERATIONS OF BLOCK READER AND MAY BE SIGNIFICANTLY SLOWER THAN THE OTHER BLOCK READERS.
     *  THIS METHOD IS ONLY IMPLEMENTED FOR CALLER CONVENIENCE AND IS NOT RECOMMENDED FOR FAST I/O
     *  NEEDS.
     */
    if (!atEOF()) {
        _raster.getBlock(buffer, _xidx*xskip, _yidx*yskip, band);
        operator++();
    } else {
        cout << "In RasterBlockIter::getNext() - Iterator has reached EOF." << endl;
    }
}

template<typename T>
void RasterBlockIter::getNext(vector<vector<T>> &buffer) {
    /*
     *  Gets the current block pointed to by the _xidx/_yidx indices and increments the iterator.
     *  Please refer to getNext(vector<vector<T>>&,size_t) above for implementation notes. Points to
     *  the default image band.
     */
    getNext(buffer, 1);
}

template<typename T>
void RasterBlockIter::setNext(T *buffer, size_t length, size_t width, size_t band) {
    /*
     *  Sets the current block pointed to by _xidx/_yidx and increments the iterator. Since this
     *  uses an arbitrary memory buffer, we need to know the dimensions of the buffer being passed
     *  in (using length/width). Points to a specific band in the image.
     */
    // Check to see if we have the off-the-end iterator
    if (!atEOF()) {
        // Read block pointed to by the indices by using the _xidx/_yidx indices and the xskip/yskip
        // block offsets
        _raster.setBlock(buffer, _xidx*xskip, _yidx*yskip, length, width, band);
        operator++();
    } else {
        cout << "In RasterBlockIter::setNext() - Iterator has reached EOF." << endl;
    }
}

template<typename T>
void RasterBlockIter::setNext(T *buffer, size_t length, size_t width) {
    /*
     *  Sets the current block pointed to by the _xidx/_yidx indices and increments the iterator.
     *  Needs to know the size of the buffer since it's a raw pointer. Points to the default band.
     */
    setNext(buffer, length, width, 1);
}

template<typename T>
void RasterBlockIter::setNext(vector<T> &buffer, size_t length, size_t width, size_t band) {
    /*
     *  Sets the current block pointed to by the _xidx/_yidx indices and increments the iterator.
     *  Needs to know the size of the buffer because it's a linearly-contiguous 1D STL container, so
     *  we can't infer the data layout. Points to a specific image band.
     */
    setNext(buffer.data(), length, width, band);
}

template<typename T>
void RasterBlockIter::setNext(vector<T> &buffer, size_t length, size_t width) {
    /*
     *  Sets the current block pointed to by the _xidx/_yidx indices and increments the iterator. As
     *  above, needs to know the size of the buffer. Points to the default image band.
     */
    setNext(buffer.data(), length, width, 1);
}

template<typename T>
void RasterBlockIter::setNext(vector<vector<T>> &buffer, size_t band) {
    /*
     *  Sets the current block pointed to by the _xidx/_yidx indices and increments the iterator. In
     *  this case of using a 2D STL container buffer, we need to implement the matching setBlock()
     *  that really is a line reader. Points to a specific image band.
     *
     *  NOTE: AS WITH RASTER::GETBLOCK(CONTAINER<CONTAINER<T>>) THIS IMPLEMENTATION ACTUALLY CALLS
     *  ITERATIONS OF BLOCK READER AND MAY BE SIGNIFICANTLY SLOWER THAN THE OTHER BLOCK READERS.
     *  THIS METHOD IS ONLY IMPLEMENTED FOR CALLER CONVENIENCE AND IS NOT RECOMMENDED FOR FAST I/O
     *  NEEDS.
     */
    if (!atEOF()) {
        _raster.setBlock(buffer, _xidx*xskip, _yidx*yskip, band);
        operator++();
    } else {
        cout << "In RasterBlockIter::setNext() - Iterator has reached EOF." << endl;
    }
}

template<typename T>
void RasterBlockIter::setNext(vector<vector<T>> &buffer) {
    /*
     *  Sets the current block pointed to by the _xidx/_yidx indices and increments the iterator.
     *  Please refer to setNext(vector<vector<T>>&,size_t) above for implementation notes. Points to
     *  the default image band.
     */
    setNext(buffer, 1);
}
