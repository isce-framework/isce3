//-*- C++ -*-
//-*- coding: utf-8 -*-


#include <iostream>
#include <cmath>
#include <vector>
#include "isce/core/LUT2d.h"
#include "gtest/gtest.h"


struct LUT2DTest : public ::testing::Test {
    virtual void SetUp() {
        fails = 0;
    }
    virtual void TearDown() {
        if (fails > 0) {
            std::cerr << "LUT2D::TearDown sees failures" << std::endl;
        }
    }
    unsigned fails;
};

TEST_F(LUT2DTest, Lookup) {

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
    for (size_t i = 0; i < N; ++i) {
        double yi = i / (N - 2.0);
        for (size_t j = 0; j < N; ++j) {
            double xj = j / (N - 2.0);
            double interp_value = lut.eval(yi, xj);
            EXPECT_NEAR(interp_value, ref_values[i][j], tol);
        }
    }

    // Clear vectors
    x_index.clear();
    y_index.clear();
    values[0].clear();
    values[1].clear();
    for (size_t i = 0; i < N; ++i)
        ref_values[i].clear();

    fails += ::testing::Test::HasFailure();

}


int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

// end of file
