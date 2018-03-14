//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_DATAIO_RASTERLINEITER_H__
#define __ISCE_DATAIO_RASTERLINEITER_H__

#include <algorithm>
#include <vector>
#include "Raster.h"

namespace isce {
  namespace dataio {
    class RasterLineIter {

    public:
      // RasterLineIter should *only* be generated from a valid Raster object (otherwise we have
        // to deal with checking for nullptr _raster values)
        RasterLineIter() = delete;
        RasterLineIter(const RasterLineIter &rli) : _raster(rli._raster), _lineidx(rli._lineidx) {}
        RasterLineIter(const Raster &r) : _raster(r), _lineidx(0) {}

        inline RasterLineIter& operator=(const RasterLineIter&);
        inline RasterLineIter& operator+=(const size_t);
        inline RasterLineIter& operator-=(const size_t);
        inline RasterLineIter& operator+(const size_t rhs) { return RasterLineIter(*this) += rhs; }
        inline RasterLineIter& operator-(const size_t rhs) { return RasterLineIter(*this) -= rhs; }
        // Prefix increment
        inline RasterLineIter& operator++();
        // Postfix increment
        inline RasterLineIter operator++(int);
        // Prefix decrement
        inline RasterLineIter& operator--();
        // Postfix decrement
        inline RasterLineIter operator--(int);
        inline bool operator==(const RasterLineIter&);
        inline bool operator!=(const RasterLineIter &rhs) { return !(*this == rhs); }

        inline void rewind() { _lineidx = 0; }
        inline void ffwd() { _lineidx = _raster.length(); }
        inline bool atEOF() { return _lineidx == _raster.length(); }
        inline RasterLineIter atBeginning();
        inline RasterLineIter atEnd();

        // (buffer, nelem, [band])
        template<typename T> void getNext(T*,size_t,size_t);
        template<typename T> void getNext(T*,size_t);
        // (buffer, [band])
        template<typename T> void getNext(std::vector<T>&,size_t);
        template<typename T> void getNext(std::vector<T>&);

        // (buffer, nelem, [band])
        template<typename T> void setNext(T*,size_t,size_t);
        template<typename T> void setNext(T*,size_t);
        // (buffer, [band])
        template<typename T> void setNext(std::vector<T>&,size_t);
        template<typename T> void setNext(std::vector<T>&);

    private:
	Raster _raster;
        size_t _lineidx;

    };

    inline RasterLineIter& RasterLineIter::operator=(const RasterLineIter &rhs) {
        _raster = rhs._raster;
        _lineidx = rhs._lineidx;
        return *this;
    }

    inline RasterLineIter& RasterLineIter::operator+=(const size_t rhs) {
        _lineidx = std::min(_lineidx+rhs, _raster.length());
        return *this;
    }

    inline RasterLineIter& RasterLineIter::operator-=(const size_t rhs) {
        // Need to account for if rhs > _lineidx (since they're size_t)
        _lineidx = (rhs > _lineidx) ? 0 : (_lineidx - rhs);
        return *this;
    }

    inline RasterLineIter& RasterLineIter::operator++() {
        _lineidx = (_lineidx == _raster.length()) ? _lineidx : (_lineidx + 1);
        return *this;
    }

    inline RasterLineIter RasterLineIter::operator++(int) {
        RasterLineIter old(*this);
        operator++();
        return old;
    }

    inline RasterLineIter& RasterLineIter::operator--() {
        _lineidx = (_lineidx == 0) ? _lineidx : (_lineidx - 1);
        return *this;
    }

    inline RasterLineIter RasterLineIter::operator--(int) {
        RasterLineIter old(*this);
        operator--();
        return old;
    }

    inline bool RasterLineIter::operator==(const RasterLineIter &rhs) {
        return (_raster.dataset == rhs._raster.dataset) && (_lineidx == rhs._lineidx); 
    }

    inline RasterLineIter RasterLineIter::atBeginning() {
        // This and atEnd() are not truly begin()/end(), so slightly different naming
        RasterLineIter ret(*this);
        ret.rewind();
        return ret;
    }

    inline RasterLineIter RasterLineIter::atEnd() {
        RasterLineIter ret(*this);
        ret.ffwd();
        return ret;
    }
}}

#define ISCE_DATAIO_RASTERLINEITER_ICC
#include "RasterLineIter.icc"
#undef ISCE_DATAIO_RASTERLINEITER_ICC

#endif
