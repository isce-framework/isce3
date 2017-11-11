//-*- C++ -*-
//-*- coding: utf-8 -*-


#include <iostream>
#include <cmath>
#include <vector>
#include "isce/core/LUT2d.h"

void testLUT() {

    // The LUT indices
    std::vector<double> x_index{0.0, 1.0, 2.0};
    std::vector<double> y_index{0.0, 1.0, 2.0};

    // The LUT values
    std::vector<std::vector<double>> values{
        {0.0, 1.0, 1.25},
        {1.0, 0.5, -0.2},
        {2.0, 0.4, -0.1}
    };

    // Initialize LUT
    isce::core::LUT2d<double> lut(x_index, y_index, values);

    // Set tolerance for comparison
    const double tol = 1.0e-14;

    // The correct values for the interpolation test
    std::vector<std::vector<double>> ref_values{
        {0.0, 0.5, 1.0, 1.125},
        {0.5, 0.625, 0.75, 0.6375},
        {1.0, 0.75, 0.5, 0.15},
        {1.5, 0.975, 0.45, 0.15}
    };
    const size_t N = ref_values.size();

    // Interpolate N values in x and y
    bool stat = true;
    for (size_t i = 0; i < N; ++i) {
        double yi = i / (N - 2.0);
        for (size_t j = 0; j < N; ++j) {
            double xj = j / (N - 2.0);
            double interp_value = lut.eval(yi, xj);
            stat = stat & (std::abs(interp_value - ref_values[i][j]) < tol);
        }
    }

    std::cout << "\n[LUT2D] ";
    if (stat) 
        std::cout << "PASSED";
    else
        std::cout << "FAILED";
    std::cout << std::endl << std::endl;

    // Clear vectors
    x_index.clear();
    y_index.clear();
    values[0].clear();
    values[1].clear();
    for (size_t i = 0; i < N; ++i)
        ref_values[i].clear();

}


int main() {

    testLUT();

    return 0;
}

// end of file
