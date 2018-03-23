// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2018
//

#ifndef ISCE_CORE_TILE_H
#define ISCE_CORE_TILE_H

#include <complex>
#include <valarray>

// Declarations
namespace isce {
    namespace core {
        template <typename T> class Tile;
    }
}

// Definition
template <typename T>
class isce::core::Tile {
    
    public:
        // Constructors
        inline Tile();
        inline Tile(const Tile &);

        // Getters for geometry
        inline size_t length() const;
        inline size_t width() const;
        inline size_t rowStart() const;
        inline size_t rowEnd() const;
        inline size_t firstImageRow() const;
        inline size_t lastImageRow() const;

        // Setters for geometry
        inline void width(size_t);
        inline void rowStart(size_t);
        inline void rowEnd(size_t);
        inline void firstImageRow(size_t);
        inline void lastImageRow(size_t);

        // Allocate memory
        inline void allocate();

        // Overload subscript operators to access valarray data
        T & operator[](size_t index) {return _data[index];}
        const T & operator[](size_t index) const {return _data[index];}

    private:
        // Geometry
        size_t _width, _rowStart, _rowEnd;
        size_t _firstImageRow, _lastImageRow;

        // Data
        std::valarray<T> _data;
};

// Get inline implementations of Tile
#define ISCE_CORE_TILE_ICC
#include "Tile.icc"
#undef ISCE_CORE_TILE_ICC

#endif

// end of file
