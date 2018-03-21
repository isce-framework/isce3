// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2018
//

#ifndef ISCE_CORE_TILE_H
#define ISCE_CORE_TILE_H

#include <complex>

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
        // Geometry
        size_t length;
        size_t width;
        size_t rowStart;
        size_t rowEnd;

        // Constructor
        inline Tile();
        // Destructor
        inline ~Tile();

        // Allocate memory
        inline void allocate();
        // Clear memory
        inline void clear();

    private:
        // Data
        T * _data;
};

// Get inline implementations of Tile
#define ISCE_CORE_TILE_ICC
#include "Tile.icc"
#undef ISCE_CORE_TILE_ICC

#endif

// end of file
