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

// pyre
#include <pyre/journal.h>

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
        inline int length() const;
        inline int width() const;
        inline int rowStart() const;
        inline int rowEnd() const;
        inline int firstImageRow() const;
        inline int lastImageRow() const;

        // Setters for geometry
        inline void width(int);
        inline void rowStart(int);
        inline void rowEnd(int);
        inline void firstImageRow(int);
        inline void lastImageRow(int);

        // Allocate memory
        inline void allocate();

        // Print out relevant attributes
        inline void declare(pyre::journal::info_t &) const;

        // Overload subscript operator to access valarray data
        T & operator[](size_t index) {return _data[index];}
        // Read-only subscript operator
        const T & operator[](size_t index) const {return _data[index];}

        // Overload () operator for 2D access
        T & operator()(size_t row, size_t col) {return _data[row*_width+col];}
        // Read-only () operator for 2D access
        const T & operator()(size_t row, size_t col) const {return _data[row*_width+col];}

        // Get reference to underlying data
        inline std::valarray<T> & data();

    private:
        // Geometry
        int _width, _rowStart, _rowEnd, _firstImageRow, _lastImageRow;

        // Data
        std::valarray<T> _data;
};

// Get inline implementations of Tile
#define ISCE_CORE_TILE_ICC
#include "Tile.icc"
#undef ISCE_CORE_TILE_ICC

#endif

// end of file
