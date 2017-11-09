//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017

#include <complex>
#include <vector>
#include "LUT2d.h"
using isce::core::LUT2d;
using std::complex;
using std::vector;

template <typename T>
T LUT2d<T>::eval(double y, double x) const {
    /*
     * Evaluate the LUT at the given indices. Note that because we've template-bound the class-type,
     * not the function-type, we don't need to forward-declare the compatible types!
     */
    size_t i0, i1, j0, j1, i, j;
    // Iterate over x indices to find x bounds
    double xdiff = -100.0;
    for (j = 0; j < x_index.size()-1; ++j) {
        // Compute difference with current x value
        xdiff = x_index[j] - x;
        // Break if sign has changed
        if (xdiff > 0.) break;
    }
    // Do the same process for finding y bounds
    double ydiff = -100.0;
    for (i = 0; i < y_index.size()-1; ++i) {
        ydiff = y_index[i] - y;
        if (ydiff > 0.0) break;
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
    T fx1 = (((x2 - x) / (x2 - x1)) * values[i0][j0]) + (((x - x1) / (x2 - x1)) * values[i0][j1]);
    T fx2 = (((x2 - x) / (x2 - x1)) * values[i1][j0]) + (((x - x1) / (x2 - x1)) * values[i1][j1]);
    // Interpolate in the y direction
    T result = (((y2 - y) / (y2 - y1)) * fx1) + (((y - y1) / (y2 - y1)) * fx2);
    return result;
}

// Forward declare the compatible types
template double LUT2d<double>::eval(double,double) const;
template complex<double> LUT2d<complex<double>>::eval(double,double) const;
