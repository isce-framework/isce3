//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#ifndef ISCE_CORE_MATRIX_H
#define ISCE_CORE_MATRIX_H

#include <cmath>
#include <valarray>
#include <vector>
#include <pyre/grid.h>

// Declaration
namespace isce {
    namespace core {
        template <typename cell_t> class Matrix;
    }
}

// isce::core::Matrix definition
template <typename cell_t>
class isce::core::Matrix {

    public:
        // Types for interfacing with pyre::grid
        using rep_t = std::array<size_t, 2>;
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
        inline Matrix();

        /** Constructor with number of rows and number of columns */
        inline Matrix(size_t nrows, size_t ncols);
        
        /** Deep copy constructor from another matrix - allocates memory and copies values */
        inline Matrix(const Matrix<cell_t> & m);

        /** Shallow copy constructor from another matrix - does not allocate own memory */
        inline Matrix(Matrix<cell_t> & m);

        /** Copy constructor from a grid view (copy values) */
        inline Matrix(const view_t & view);

        /** Shallow copy constructor from raw pointer to data - does not allocate own memory */
        inline Matrix(cell_t * data, size_t nrows, size_t ncols);

        /** Shallow copy constructor from an std::valarray - does not allocate own memory */
        inline Matrix(std::valarray<cell_t> & data, size_t ncols);

        /** Shallow copy constructor from an std::vector - does not allocate own memory */
        inline Matrix(std::vector<cell_t> & data, size_t ncols);

        /** Destructor */
        inline ~Matrix();

        /** Deep assignment operator - allocates memory and copies values */
        inline Matrix<cell_t> & operator=(const Matrix<cell_t> & m);

        /** Shallow assignment operator - does not allocate own memory */
        inline Matrix<cell_t> & operator=(Matrix<cell_t> & m);
        
        /** Assignment operator from a grid view (copy values) */
        inline Matrix<cell_t> & operator=(const view_t & view);

        /** Resize memory for a given number of rows and columns */
        inline void resize(size_t nrows, size_t ncols);

        /** Extract copy of sub-matrix given starting indices and span */
        inline Matrix<cell_t> submat(size_t row, size_t col, size_t rowspan, size_t colspan);
            
        /** Access to data buffer */
        inline cell_t * data();

        /** Read-only access to data buffer */
        inline const cell_t * data() const;

        /** Access matrix value for a given row and column */
        inline cell_t & operator()(size_t row, size_t col);

        /** Read-only access to matrix value for a given row and column */
        inline const cell_t & operator()(size_t row, size_t col) const;

        /** Access matrix value for a flattened index */
        inline cell_t & operator()(size_t index);

        /** Read-only access to matrix value for a flattened index */
        inline const cell_t & operator()(size_t index) const;

        /** Access matrix value for a given grid::index_type */
        inline cell_t & operator[](const index_t & index);

        /** Read-only access to matrix value for a given grid::idnex_type */
        inline const cell_t & operator[](const index_t & index) const;

        /** Get shape information as grid::shape_type */
        inline shape_t shape() const;

        /** Get matrix width */
        inline size_t width() const;

        /** Get matrix length */
        inline size_t length() const;

    private:
        // Data members
        size_t _nrows;
        size_t _ncols;
        cell_t * _buffer;
        bool _owner;
};

// Get inline implementations for Matrix
#define ISCE_CORE_MATRIX_ICC
#include "Matrix.icc"
#undef ISCE_CORE_MATRIX_ICC

#endif

// end of file
