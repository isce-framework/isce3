#ifndef ISCE_IMAGE_TILE_H
#define ISCE_IMAGE_TILE_H
#pragma once

#include "forward.h"

#include <complex>
#include <valarray>

// pyre
#include <pyre/journal.h>

// Definition
template <typename T>
class isce3::image::Tile {

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

        // Print out relevant attributes
        inline void declare(pyre::journal::info_t &) const;

        // Overload subscript operator to access valarray data
        inline T & operator[](size_t index) {return _data[index];}
        // Read-only subscript operator
        inline const T & operator[](size_t index) const {return _data[index];}

        // Overload () operator for 2D access
        inline T & operator()(size_t row, size_t col) {return _data[row*_width+col];}

        // Read-only () operator for 2D access
        inline const T & operator()(size_t row, size_t col) const {return _data[row*_width+col];}

        // Get reference to underlying data
        inline std::valarray<T> & data();

    private:
        // Geometry
        size_t _width;

        // First row of original tile without buffer needed to account for
        // offset and half of chip size. Defined w.r.t. source raster.
        size_t _rowStart;

        // Last row of original tile without buffer needed to account for
        // offset and half of chip size. Defined w.r.t. source raster.
        size_t _rowEnd;

        // First row of original tile to read from. Includes buffer needed to
        // account for offset and half of chip size. Defined w.r.t. source raster.
        size_t _firstImageRow;

        // Last row of original tile to read from. Includes buffer needed to
        // account for offset and half of chip size. Defined w.r.t. source raster.
        size_t _lastImageRow;

        // Data
        std::valarray<T> _data;
};

// Get inline implementations of Tile
#define ISCE_IMAGE_TILE_ICC
#include "Tile.icc"
#undef ISCE_IMAGE_TILE_ICC

#endif
