//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#pragma once

#include "forward.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <valarray>
#include <vector>

#include "EMatrix.h"

/** Data structure for a 2D row-major matrix*/
template <typename T>
class isce3::core::Matrix :
    public Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
{

        using super_t = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    public:
        using index_t = typename super_t::Index;

    public:
        /** Default constructor */
        Matrix() : super_t(0, 0) {}

        Matrix(index_t rows, index_t cols) : super_t(rows, cols) {}

        template<typename Derived>
        Matrix(const Eigen::Block<Derived>& b) : super_t(b) {}

        /** Copy constructor from raw pointer to data */
        Matrix(T * data, size_t nrows, size_t ncols) :
            super_t(Eigen::Map<super_t>(data, nrows, ncols))
        {
            assert(ncols <= std::numeric_limits<Eigen::Index>::max());
            assert(nrows <= std::numeric_limits<Eigen::Index>::max());
        }

        /** Copy constructor from an std::valarray */
        Matrix(std::valarray<T> & data, size_t ncols) :
            super_t(Eigen::Map<super_t>(data.data(), data.size() / ncols, ncols))
        {
            assert(ncols <= std::numeric_limits<Eigen::Index>::max());
        }

        /** Copy constructor from an std::vector */
        Matrix(std::vector<T> & data, size_t ncols) :
            super_t(Eigen::Map<super_t>(data.data(), data.size() / ncols, ncols))
        {
            assert(ncols <= std::numeric_limits<Eigen::Index>::max());
        }

        /** Extract copy of sub-matrix given starting indices and span */
        auto submat(size_t row, size_t col, size_t rowspan, size_t colspan) {
            assert(col <= std::numeric_limits<Eigen::Index>::max());
            assert(row <= std::numeric_limits<Eigen::Index>::max());
            assert(colspan <= std::numeric_limits<Eigen::Index>::max());
            assert(rowspan <= std::numeric_limits<Eigen::Index>::max());
            return this->block(row, col, rowspan, colspan);
        }

        /** Fill with zeros */
        void zeros() { this->fill(0); }

        /** Get matrix width */
        size_t width() const { return this->cols(); }

        /** Get matrix length */
        size_t length() const { return this->rows(); }

        auto map() const {
            return Eigen::Map<const super_t> {
                    this->data(),
                    this->rows(),
                    this->cols(),
            };
        }

};
