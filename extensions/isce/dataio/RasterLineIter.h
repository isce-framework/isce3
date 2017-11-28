//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_DATAIO_RASTERLINEITER_H__
#define __ISCE_DATAIO_RASTERLINEITER_H__

#include "Raster.h"

namespace isce { namespace dataio {
    template<typename T>
    struct RasterLineIter {
        Raster &_raster;
        T *_linebuf;
        // Index of current line the iterator points to
        size_t _lineidx;
        // Index of current line the buffer is holding
        size_t _linebufidx;
        size_t _linewidth;
        size_t _rasterband;
        bool _readonly;

        // It makes no sense to have a default constructor since we're storing Raster as a
        // reference
        RasterLineIter() = delete;
        // Note that copy-constructing an iterator doesn't explicitly copy the old buffered line
        // into the new iterator (since we want to hold to only reading data when explicitly
        // asked for). The copy-constructor simply allocates space for the new line (much faster
        // than outright reading the line in). Setting the _lineidx to be different than the
        // _linebufidx means that dereferencing will trigger reading in the required line.
        RasterLineIter(const RasterLineIter<T> &rli) : _raster(rli._raster),
                                                       _linebuf(new T[rli._linewidth]),
                                                       _lineidx(rli._lineidx),
                                                       _linebufidx(rli._lineidx-1),
                                                       _linewidth(rli._linewidth),
                                                       _rasterband(rli._rasterband),
                                                       _readonly(rli._readonly) {}
        // Constructor from Raster. Initialize _linebufidx to be different than _lineidx to
        // trigger an i/o read when accessing a line.
        RasterLineIter(const Raster &r, size_t b=1) : _raster(r), _linebuf(new T[r.getWidth()]), 
                                                      _lineidx(0), _linebufidx(1), 
                                                      _linewidth(r.getWidth()), _rasterband(b),
                                                      _readonly(r._readonly)
        ~RasterLineIter() { delete[] _linebuf; }
        
        // Cannot implement operator= while using internal Raster storage as a reference
        RasterLineIter<T>& operator=(const RasterLineIter<T>&) = delete;
        inline RasterLineIter<T>& operator+=(size_t);
        inline RasterLineIter<T>& operator-=(size_t);
        inline RasterLineIter<T>& operator+(size_t rhs) { return RasterLineIter(*this) += rhs; }
        inline RasterLineIter<T>& operator-(size_t rhs) { return RasterLineIter(*this) -= rhs; }
        // Prefix operator++
        inline RasterLineIter<T>& operator++();
        // Postfix operator++
        inline RasterLineIter<T> operator++(int);
        // Prefix operator--
        inline RasterLineIter<T>& operator--();
        // Postfix operator--
        inline RasterLineIter<T> operator--(int);
        inline bool operator==(const RasterLineIter<T>&) { return _raster == rhs._raster; }
        inline bool operator!=(const RasterLineIter<T>&) { return !(*this == rhs); }
        // Dereference as an rvalue
        const T* operator*();
        // Note: Random-access uses a temporary iterator, so as with iterators generally using
        // this repeatedly is not advised (since it will always need to read the random line; does
        // not leverage the buffer)
        inline const T* operator[](size_t n) { return *(*this + n); }

        const T* getLineSequential();
        void setLineSequential(const T*);
    };

    template<typename T> 
    inline RasterLineIter<T>& RasterLineIter<T>::operator+=(size_t rhs) {
        if ((_lineidx + n) < _raster.getLength()) _lineidx += n;
        else _lineidx = (_lineidx + n) - _raster.getLength();
        return *this;
    }

    template<typename T> 
    inline RasterLineIter<T>& RasterLineIter<T>::operator-=(size_t rhs) {
        if (n <= _lineidx) _lineidx -= n;
        else _lineidx = (_lineidx + _raster.getLength()) - n;
        return *this;
    }

    template<typename T>
    inline RasterLineIter<T>& RasterLineIter<T>::operator++() {
        if (_lineidx < (_raster.getLength()-1)) _lineidx++;
        else _lineidx = 0;
        return *this;
    }

    template<typename T>
    inline RasterLineIter<T> RasterLineIter<T>::operator++(int) {
        RasterLineIter old(*this);
        operator++();
        return old;
    }

    template<typename T>
    inline RasterLineIter<T>& RasterLineIter<T>::operator--() {
        if (_lineidx > 0) _lineidx--;
        else _lineidx = _raster.getLength() - 1;
        return *this;
    }

    template<typename T>
    inline RasterLineIter<T> RasterLineIter<T>::operator--(int) {
        RasterLineIter old(*this);
        operator--();
        return old;
    }
}}

#endif
