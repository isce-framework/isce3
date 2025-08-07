//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018
//

#include <cmath>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"

// isce3::core
#include "isce3/core/Constants.h"
#include "isce3/core/Utilities.h"
#include "isce3/core/Matrix.h"

TEST(MatrixTest, SimpleConstructor) {
    // Make a matrix with a fixed shape
    isce3::core::Matrix<double> M(3, 3);
    ASSERT_EQ(M.width(), 3);
    ASSERT_EQ(M.length(), 3);
}

TEST(MatrixTest, Resize) {
    // Make a matrix with a fixed shape
    isce3::core::Matrix<double> M(3, 3);
    // Resize it
    M.resize(5, 5);
    // Check shape 
    ASSERT_EQ(M.width(), 5);
    ASSERT_EQ(M.length(), 5);
}

TEST(MatrixTest, FixedValues) {
    // Make a matrix with a fixed shape
    isce3::core::Matrix<double> M(3, 3);

    // Fill it with zeros and check values
    M.zeros();
    for (size_t count = 0; count < (M.width() * M.length()); ++count) {
        ASSERT_NEAR(M(count), 0.0, 1.0e-12);
    }

    // Fill it with a constant value and check
    M.fill(10.0);
    for (size_t count = 0; count < (M.width() * M.length()); ++count) {
        ASSERT_NEAR(M(count), 10.0, 1.0e-12);
    }
}

TEST(MatrixTest, VectorConstructor) {
    // Make a vector of values
    std::vector<double> values = isce3::core::arange(0.0, 9.0, 1.0);
    // Make a matrix from the vector
    isce3::core::Matrix<double> M(values, 3);

    // Check the shape
    ASSERT_EQ(M.width(), 3);
    ASSERT_EQ(M.length(), 3);

    // Check the values with flattened indices
    for (size_t i = 0; i < values.size(); ++i) {
        // Get the matrix value
        const double mat_val = M(i);
        // Get the vector value
        const double vec_val = values[i];
        // Check
        ASSERT_NEAR(mat_val, vec_val, 1.0e-12);
    }
}

TEST(MatrixTest, CopyConstructor) {
    // Make a vector of values 
    std::vector<double> values = isce3::core::arange(0.0, 9.0, 1.0);
    // Make a const matrix from the vector
    const isce3::core::Matrix<double> M(values, 3);
    // Make a deep copy (by passing in const matrix)
    isce3::core::Matrix<double> N(M);

    // Change value of middle element of copied matrix
    N(1, 1) = 20.0;
    // Check corresponding value in original matrix has NOT been updated
    ASSERT_NEAR(M(1, 1), 4.0, 1.0e-12);
}

TEST(MatrixTest, MatrixView) {
    // Make a vector of values 
    std::vector<double> values = isce3::core::arange(0.0, 9.0, 1.0);
    // Make a matrix from the vector
    isce3::core::Matrix<double> M(values, 3);
    // Get a view of a subset of the matrix
    auto view = M.submat(1, 1, 2, 2);

    // Vector of expected values
    std::vector<double> expected{4.0, 5.0, 7.0, 8.0};

    // Compare values
    size_t count = 0;
    for (int row = 0; row < view.rows(); row++) {
        for (int col = 0; col < view.cols(); col++) {
            double view_val = view(row, col);
            ASSERT_NEAR(view_val, expected[count], 1.0e-12);
            ++count;
        }
    }
}

TEST(MatrixTest, MatrixViewConstructor) {
    // Make a vector of values 
    std::vector<double> values = isce3::core::arange(0.0, 9.0, 1.0);
    // Make a matrix from the vector
    isce3::core::Matrix<double> M(values, 3);
    // Construct a new matrix from a view of a subset of the matrix
    isce3::core::Matrix<double>N(M.submat(1, 1, 2, 2));

    // Vector of expected values
    std::vector<double> expected{4.0, 5.0, 7.0, 8.0};

    // Check shape
    ASSERT_EQ(N.width(), 2);
    ASSERT_EQ(N.length(), 2);

    // Compare values
    for (size_t count = 0; count < (N.width() * N.length()); ++count) {
        ASSERT_NEAR(N(count), expected[count], 1.0e-12);
    }
}

TEST(MatrixTest, MatrixViewSet) {
    // Make a vector of values 
    std::vector<double> values = isce3::core::arange(0.0, 9.0, 1.0);
    // Make a matrix from the vector
    isce3::core::Matrix<double> M(values, 3);

    // Make a new matrix of zeros
    isce3::core::Matrix<double> N(3, 3);
    N.zeros();

    // Set column of matrix with row from original matrix
    N.submat(0, 1, 3, 1) = M.submat(1, 0, 1, 3).transpose();

    // Vector of expected values
    std::vector<double> expected{0.0, 3.0, 0.0,
                                 0.0, 4.0, 0.0,
                                 0.0, 5.0, 0.0};

    // Compare values
    for (int row = 0; row < N.rows(); row++) {
        for (int col = 0; col < N.cols(); col++) {
            ASSERT_NEAR(N(row, col), expected[row * N.cols() + col], 1.0e-12);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
