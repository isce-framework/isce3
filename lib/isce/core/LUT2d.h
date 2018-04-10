//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CORE_LUT2D_H
#define ISCE_CORE_LUT2D_H

#include <complex>
#include <vector>

// Declaration
namespace isce {
    namespace core {
        template <typename T>
        struct LUT2d;
    }
}

// LUT2d declaration
template <typename T>
struct isce::core::LUT2d {
    // Vectors to hold indices in both dimensions
    std::vector<double> x_index, y_index;
    // 2D vector to hold values
    std::vector<std::vector<T>> values;

    LUT2d() = default;
    LUT2d(std::vector<double> &_xidx, std::vector<double> &_yidx, 
          std::vector<std::vector<T>> &_vals) : x_index(_xidx), y_index(_yidx), values(_vals) {}
    T eval(double, double) const;
};

// Forward declarations for the constructor
//template LUT2d<double>::LUT2d(std::vector<double>&,std::vector<double>&,
//                              std::vector<std::vector<double>>&);
//template LUT2d<std::complex<double>>::LUT2d(std::vector<double>&,std::vector<double>&,
//                                            std::vector<std::vector<std::complex<double>>>&);

#endif
