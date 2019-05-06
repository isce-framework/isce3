//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2017-2019

#ifndef ISCE_CORE_CUBE_H
#define ISCE_CORE_CUBE_H

#include <cmath>
#include <valarray>
#include <vector>
#include <pyre/grid.h>

// Declaration
namespace isce {
    namespace core {
        template <typename cell_t> class Cube;
    }
}

// isce::core::Cube definition
template <typename cell_t>
class isce::core::Cube {

    public:
        // Types for interfacing with pyre::grid
        using rep_t = std::array<size_t, 3>;
        using index_t = pyre::grid::index_t<rep_t>;
        using layout_t = pyre::grid::layout_t<index_t>;

        // Use a grid with view memory storage
        using grid_t = pyre::grid::grid_t<cell_t, layout_t, pyre::memory::view_t<cell_t>>;

        // Dependent types
        using view_t = typename grid_t::view_type;
        using shape_t = typename layout_t::shape_type;
        using slice_t = typename layout_t::slice_type;
        using packing_t = typename layout_t::packing_type;

    public:
        /** Default constructor */
        inline Cube();

        /** Constructor with number of slices, rows and columns */
        inline Cube(size_t nslices, size_t nrows, size_t ncols);
        
        /** Deep copy constructor from another cube - allocates memory and copies values */
        inline Cube(const Cube<cell_t> & m);

        /** Shallow copy constructor from another cube - does not allocate own memory */
        inline Cube(Cube<cell_t> & m);

        /** Shallow copy constructor from raw pointer to data - does not allocate own memory */
        inline Cube(cell_t * data, size_t nslices, size_t nrows, size_t ncols);

        /** Shallow copy constructor from an std::valarray - does not allocate own memory */
        inline Cube(std::valarray<cell_t> & data, size_t nrows, size_t ncols);

        /** Shallow copy constructor from an std::vector - does not allocate own memory */
        inline Cube(std::vector<cell_t> & data, size_t nrows, size_t ncols);

        /** Destructor */
        inline ~Cube();

        /** Deep assignment operator - allocates memory and copies values */
        inline Cube<cell_t> & operator=(const Cube<cell_t> & m);

        /** Shallow assignment operator - does not allocate own memory */
        inline Cube<cell_t> & operator=(Cube<cell_t> & m);
        
        /** Resize memory for a given number of slices, rows and columns */
        inline void resize(size_t nslices, size_t nrows, size_t ncols);

        /** Extract copy of sub-cube given starting indices and span */
        inline const view_t subcube(size_t slice, size_t row, size_t col, size_t slicespan, size_t rowspan, size_t colspan);
            
        /** Access to data buffer */
        inline cell_t * data();

        /** Read-only access to data buffer */
        inline const cell_t * data() const;

        /** Access to data buffer at a specific slice */
        inline cell_t * sliceptr(size_t slice);

        /** Read-only access to data buffer at a specific slice */
        inline const cell_t * sliceptr(size_t slice) const;

        /** Access matrix value for a given slice, row and column */
        inline cell_t & operator()(size_t slice, size_t row, size_t col);

        /** Read-only access to matrix value for a given slice, row and column */
        inline const cell_t & operator()(size_t slice, size_t row, size_t col) const;

        /** Access matrix value for a flattened index */
        inline cell_t & operator()(size_t index);

        /** Read-only access to matrix value for a flattened index */
        inline const cell_t & operator()(size_t index) const;

        /** Access matrix value for a given grid::index_type */
        inline cell_t & operator[](const index_t & index);

        /** Read-only access to matrix value for a given grid::idnex_type */
        inline const cell_t & operator[](const index_t & index) const;

        /** Fill with zeros */
        inline void zeros();

        /** Fill with a constant value */
        inline void fill(cell_t value);

        /** Get shape information as grid::shape_type */
        inline shape_t shape() const;

        /** Get cube width */
        inline size_t width() const;

        /** Get cube length */
        inline size_t length() const;

        /** Get cube height */
        inline size_t height() const;

        /** Get byteoffset for row and column for reading flat binary buffer */
        inline shape_t byteoffset() const;

    // Data members
    private:
        // Shape information
        size_t _nslices;
        size_t _nrows;
        size_t _ncols;

        // Dynamic memory data
        cell_t * _buffer;
        bool _owner;

        // grid pointer for slicing support
        grid_t * _grid;

    // Utility functions
    private:
        // Reset grid pointer
        inline void _resetGrid();
};

// Get inline implementations for Cube
#define ISCE_CORE_CUBE_ICC
#include "Cube.icc"
#undef ISCE_CORE_CUBE_ICC

#endif

// end of file
