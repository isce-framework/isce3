//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017

#include <cstddef>
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>
#include "LUT2d.h"

// Constructor from vectors of indices and values
template <typename T>
isce::core::LUT2d<T>::LUT2d(std::vector<double> & x_index, std::vector<double> & y_index,
    std::vector<std::vector<T>> & values) {
    // Copy input vectors
    this->x_index = x_index;
    this->y_index = y_index;
    this->values = values;
    // Store sizes
    _xsize = x_index.size();
    _ysize = y_index.size();
}

// Forward declarations for each type
template isce::core::LUT2d<double>::LUT2d(std::vector<double> & x_index,
    std::vector<double> & y_index, std::vector<std::vector<double>> & values);
template isce::core::LUT2d<std::complex<double>>::LUT2d(std::vector<double> & x_index,
    std::vector<double> & y_index, std::vector<std::vector<std::complex<double>>> & values);


//// Method to fill LUT values from an isce raster object
//template <typename T>
//void isce::core::LUT2d<T>::setValuesFromRaster(isce::core::Raster<T> & raster) {
//
//    // Consistency checks
//    assert((size_t) raster.numRows == _ysize);
//    assert((size_t) raster.numCols == _xsize);
//
//    // Re-size values vector
//    values.resize(_ysize);
//
//    // Loop over lines to fill values
//    for (size_t i = 0; i < _ysize; ++i) {
//        // Raster reads a line of data
//        raster.getLine(i);
//        // Re-size row of values
//        values[i].resize(_xsize);
//        // Copy values from data line to internal values vector
//        for (size_t j = 0; j < _xsize; ++j) {
//            values[i][j] = raster.getValue(j);
//        }
//    }
//
//}

//// Forward declrations for each type
//template void isce::core::LUT2d<double>::setValuesFromRaster(
//    isce::core::Raster<double> & raster);
//template void isce::core::LUT2d<std::complex<double>>::setValuesFromRaster(
//    isce::core::Raster<std::complex<double>> & raster);


// Evaluation
template <typename T>
T isce::core::LUT2d<T>::eval(double y, double x) {

    size_t i0, i1, j0, j1, i, j;

    // Iterate over x indices to find x bounds
    double xdiff = -100.0;
    for (j = 0; j < _xsize - 1; ++j) {
        // Compute difference with current x value
        xdiff = x_index[j] - x;
        // Break if sign has changed
        if (xdiff > 0.0) {
            break;
        }
    }

    // Do the same process for finding y bounds
    double ydiff = -100.0;
    for (i = 0; i < _ysize - 1; ++i) {
        ydiff = y_index[i] - y;
        if (ydiff > 0.0) {
            break;
        }
    }

    // The indices of the x bounds
    j0 = j - 1;
    j1 = j; 

    // The indices of the y bounds
    i0 = i - 1;
    i1 = i;

    // Get x and y values at each corner
    double x1 = x_index[j0];
    double x2 = x_index[j1];
    double y1 = y_index[i0];
    double y2 = y_index[i1];

    // Interpolate in the x direction
    T fx1 = (x2 - x) / (x2 - x1) * values[i0][j0] + (x - x1) / (x2 - x1) * values[i0][j1];
    T fx2 = (x2 - x) / (x2 - x1) * values[i1][j0] + (x - x1) / (x2 - x1) * values[i1][j1];

    // Interpolate in the y direction
    T result = (y2 - y) / (y2 - y1) * fx1 + (y - y1) / (y2 - y1) * fx2;
    return result;

}

// Forward declarations for each type
template double isce::core::LUT2d<double>::eval(double y, double x);
template std::complex<double> isce::core::LUT2d<std::complex<double>>::eval(double y, double x);


// Utility function to save sizes of indices from their vectors
template <typename T>
void isce::core::LUT2d<T>::setDimensions() {
    _xsize = x_index.size();
    _ysize = y_index.size();
}

// Forward declarations for each type
template void isce::core::LUT2d<double>::setDimensions();
template void isce::core::LUT2d<std::complex<double>>::setDimensions();


// Destructor clears the vectors
template <typename T>
isce::core::LUT2d<T>::~LUT2d() {
    x_index.clear();
    y_index.clear();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i].clear();
    }
}

// Forward declarations for each type
template isce::core::LUT2d<double>::~LUT2d();
template isce::core::LUT2d<std::complex<double>>::~LUT2d();

// end of file
