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

        /** multi-looking an array of complex data */
        void multilook(std::valarray<std::complex<T>> &input, 
                        std::valarray<std::complex<T>> &output);

        void multilook(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output,
                        std::complex<T> noDataValue);

        void multilook(std::valarray<std::complex<T>> &input,
                        std::valarray<bool> &mask,
                        std::valarray<std::complex<T>> &output);
        
        void multilook(std::valarray<std::complex<T>> &input,
                        std::valarray<T> &weights,
                        std::valarray<std::complex<T>> &output);
        

        /** multi-looking an array of real data */
        void multilook(std::valarray<T> &input,
                        std::valarray<T> &output);

        void multilook(std::valarray<T> &input,
                        std::valarray<T> &output,
                        T noDataValue);

        void multilook(std::valarray<T> &input,
			std::valarray<bool> &mask,
                        std::valarray<T> &output);

        void multilook(std::valarray<T> &input,
                        std::valarray<T> &weights,
                        std::valarray<T> &output);
        
        /** multi-looking anmplitude of an array of complex data */
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
	size_t _method;

        T _invalidData;
};

// Get inline implementations for Looks
#define ISCE_SIGNAL_LOOKS_ICC
#include "Looks.icc"
#undef ISCE_SIGNAL_LOOKS_ICC

#endif

