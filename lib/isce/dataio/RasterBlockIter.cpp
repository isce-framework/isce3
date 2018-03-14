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
  size_t nBlockPerLine = ceil((1. * _raster.width()) / yskip());
  size_t nBlockPerCol = ceil((1. * _raster.length()) / xskip());
    // globalBlock finds the smaller between the linear index plus increment and the off-the-end
    // block (which has an index equal to the total number of blocks)
    size_t globalBlock = min((xidx() * nBlockPerLine) + yidx() + rhs, nBlockPerLine * nBlockPerCol);
    // Now that we know the new "global" index, calculate backwards to the block indices
    _xidx = globalBlock / nBlockPerLine;
    _yidx = globalBlock % nBlockPerLine;
    return *this;
}

RasterBlockIter& RasterBlockIter::operator-=(const size_t rhs) {
    /*
     *  Decrement-assignment operator. See operator+= for indexing specifics and conventions.
     */
  size_t nBlockPerLine = ceil((1. * _raster.width()) / yskip());
  size_t globalBlock = (xidx() * nBlockPerLine) + yidx();
    // Need to be careful because the indices are unsigned. Unlike operator+=, we need to use
    // slightly different logic to handle 0-boundaries
  _xidx = (rhs <= globalBlock) ? (globalBlock - rhs) / nBlockPerLine : 0;
  _yidx = (rhs <= globalBlock) ? (globalBlock - rhs) % nBlockPerLine : 0;
    return *this;
}
