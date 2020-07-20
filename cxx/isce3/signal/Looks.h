// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Bryan Riel
// Copyright 2018-
//

#pragma once

#include "forward.h"

// pyre
#include <pyre/journal.h>

#include <isce3/core/Utilities.h>
#include <isce3/io/Raster.h>

namespace isce3 {
namespace signal {

bool verifyComplexToRealCasting(isce3::io::Raster& input_raster,
                                isce3::io::Raster& output_raster, int& exponent);
}
} // namespace isce3

template<class T>
class isce3::signal::Looks {
    public:
        Looks() {};

        /** Constructor with multi-looking factor
         * @param[in] colsLooks number of looks in the
         * range direction (columns)
         * @param[in] rowsLooks number of looks in the
         * azimuth direction (rows)
         */
        inline Looks(size_t colsLooks, size_t rowsLooks);

        ~Looks() {};

        /** Multi-looking with rasters
         * @param[in] input_raster input raster
         * @param[out] output raster
         * @param[in] exponent the power to which the absolute of complex
         * data are raisen to before multi-looking. If exponent = 0,
         * the default exponent is used, i.e. exponent = 1 (linear) for
         * float-to-float or complex-to-complex multilooking; or
         * exponent = 2 (squared) for complex-to-float multilooking.
         */
        void multilook(isce3::io::Raster& input_raster,
                       isce3::io::Raster& output_raster, int exponent = 0);

        /** Multi-looking an array of real data */
        void multilook(std::valarray<T>& input, std::valarray<T>& output);

        /** Multi-looking an array of real data (excluding noData values) */
        void multilook(std::valarray<T>& input, std::valarray<T>& output,
                       T noDataValue);

        /** \brief Multi-looking an array of real data. 
         * A binary mask is used to mask the data before multilooking */
        void multilook(std::valarray<T>& input, std::valarray<bool>& mask,
                       std::valarray<T>& output);

        /** Multi-looking an array of real data (a weighted averaging)*/
        void multilook(std::valarray<T>& input, std::valarray<T>& weights,
                       std::valarray<T>& output);

        /** Multi-looking an array of complex data */
        void multilook(std::valarray<std::complex<T>>& input,
                       std::valarray<std::complex<T>>& output);

        /** Multi-looking an array of complex data (excluding noData values)*/
        void multilook(std::valarray<std::complex<T>>& input,
                       std::valarray<std::complex<T>>& output,
                       std::complex<T> noDataValue);

        /** \brief Multi-looking an array of complex data. 
         * A binary mask is used to mask data before multilooking*/
        void multilook(std::valarray<std::complex<T>>& input,
                       std::valarray<bool>& mask,
                       std::valarray<std::complex<T>>& output);

        /** \brief Multi-looking an array of complex data.
         * The complex data are weighted based on the input weight array.*/
        void multilook(std::valarray<std::complex<T>>& input,
                       std::valarray<T>& weights,
                       std::valarray<std::complex<T>>& output);

        /** Multi-looking amplitude of an array of complex data.
         * The amplitudes may be raised by an exponent before summation
         * \f[
         * a = \frac{\sum_{i=1}^{N} |x_i|^{exponent}}{N}
         * \f]
         *
         * where \f$a\f$ represents the sum of amplitudes (to the power of
         * exponent) in a window, N is the number of pixels in the multi-looking
         * window, and x_i is the ith element of the complex data.
         */
        void multilook(std::valarray<std::complex<T>>& input,
                       std::valarray<T>& output, int exponent);

        /** Set number of rows in the data before multi-looking */
        inline void nrows(int);

        /** Set number of columns in the data before multi-looking */
        inline void ncols(int);

        /** Set number of looks to be taken on rows */
        inline void rowsLooks(int);

        /** Set number of looks to be taken on columns */
        inline void colsLooks(int);

        /** Set number rows after mult-looking */
        inline void nrowsLooked(int);

        /** Set number of columns after multi-looking */
        inline void ncolsLooked(int);

    private:
        // number of columns before multilooking
        size_t _ncols;

        // number of rows before multilooking
        size_t _nrows;

        // number of columns after multilooking
        size_t _ncolsLooked;

        // number of rows after multilooking
        size_t _nrowsLooked;

        // number of looks in range direction (columns)
        size_t _colsLooks;

        // numbe of looks in azimuth direction (rows)
        size_t _rowsLooks;

        // multilooking method
        // size_t _method;
};

template<class T>
isce3::signal::Looks<T>::Looks(size_t colsLooks, size_t rowsLooks)
    : _colsLooks(colsLooks), _rowsLooks(rowsLooks) {}

// Get inline implementations for Looks
#define ISCE_SIGNAL_LOOKS_ICC
#include "Looks.icc"
#undef ISCE_SIGNAL_LOOKS_ICC
