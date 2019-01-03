// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Bryan Riel
// Copyright 2018-
//

#ifndef ISCE_LIB_LOOKS_H
#define ISCE_LIB_LOOKS_H

# include <assert.h>

// pyre
#include <portinfo>
#include <pyre/journal.h>

#include <isce/core/Utilities.h>
#include <isce/io/Raster.h>

namespace isce {
    namespace signal {
        template<class T>
        class Looks;
    }
}

template<class T>
class isce::signal::Looks {
    public:
        Looks() {};

        ~Looks() {};
    
    //void multilook(isce::io::Raster &input, isce::io::Raster &output);

        /** Multi-looking an array of real data */
        void multilook(std::valarray<T> &input,
                        std::valarray<T> &output);

         /** Multi-looking an array of real data (excluding noData values) */     
        void multilook(std::valarray<T> &input,
                        std::valarray<T> &output,
                        T noDataValue);

        /** \brief Multi-looking an array of real data. 
         * A binary mask is used to mask the data before multilooking */
        void multilook(std::valarray<T> &input,
            std::valarray<bool> &mask,
                        std::valarray<T> &output);

        /** Multi-looking an array of real data (a weighted averaging)*/
        void multilook(std::valarray<T> &input,
                        std::valarray<T> &weights,
                        std::valarray<T> &output);

        /** Multi-looking an array of complex data */
        void multilook(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);

        /** Multi-looking an array of complex data (excluding noData values)*/
        void multilook(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output,
                        std::complex<T> noDataValue);

        /** \brief Multi-looking an array of complex data. 
         * A binary mask is used to mask data before multilooking*/
        void multilook(std::valarray<std::complex<T>> &input,
                        std::valarray<bool> &mask,
                        std::valarray<std::complex<T>> &output);

        /** \brief Multi-looking an array of complex data.
         * The complex data are weighted based on the input weight array.*/
        void multilook(std::valarray<std::complex<T>> &input,
                        std::valarray<T> &weights,
                        std::valarray<std::complex<T>> &output);

        /** Multi-looking anmplitude of an array of complex data. 
        * The amplitudes may be raised by an exponent before summation
        * \f[
        * a = \sum_{i=0}^{N} |x_i|^p
        * \f]
        *
        * where \f$a\f$ represents the sum of amplitudes (to the power of p) in a widow, N is the number of pixels in the multi-looking window, x_i is the ith element of the complex data, p represents the exponent.
        */
        void multilook(std::valarray<std::complex<T>> &input,
                        std::valarray<T> &output, int p);

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

        // number of rows before multilooking
        size_t _nrows;

        // number of columns before multilooking
        size_t _ncols;

        // number of rows after multilooking
        size_t _nrowsLooked;

    // number of columns after multilooking
        size_t _ncolsLooked;

        // number of looks in range direction (columns)
        size_t _colsLooks;

        // numbe of looks in azimuth direction (rows)
        size_t _rowsLooks;

    // multilooking method
    //size_t _method;

};

// Get inline implementations for Looks
#define ISCE_SIGNAL_LOOKS_ICC
#include "Looks.icc"
#undef ISCE_SIGNAL_LOOKS_ICC

#endif

