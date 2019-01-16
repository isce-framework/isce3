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

// isce::core
#include "isce/core/Constants.h"
#include "isce/core/Utilities.h"
#include "isce/core/Matrix.h"

TEST(MatrixTest, SimpleConstructor) {
    // Make a matrix with a fixed shape
    isce::core::Matrix<double> M(3, 3);
    ASSERT_EQ(M.width(), 3);
    ASSERT_EQ(M.length(), 3);
}

TEST(MatrixTest, Resize) {
    // Make a matrix with a fixed shape
    isce::core::Matrix<double> M(3, 3);
    // Resize it
    M.resize(5, 5);
    // Check shape 
    ASSERT_EQ(M.width(), 5);
    ASSERT_EQ(M.length(), 5);
}

TEST(MatrixTest, FixedValues) {
    // Make a matrix with a fixed shape
    isce::core::Matrix<double> M(3, 3);

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
    std::vector<double> values = isce::core::arange(0.0, 9.0, 1.0);
    // Make a matrix from the vector
    isce::core::Matrix<double> M(values, 3);

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
    std::vector<double> values = isce::core::arange(0.0, 9.0, 1.0);
    // Make a matrix from the vector
    isce::core::Matrix<double> M(values, 3);
    // Make a shallow copy
    isce::core::Matrix<double> N(M);

    // Check shapes are equal
    ASSERT_EQ(M.width(), N.width());
    ASSERT_EQ(M.length(), N.length());

    // Check the values
    for (size_t i = 0; i < M.length(); ++i) {
        for (size_t j = 0; j < M.width(); ++j) {
            ASSERT_NEAR(M(i,j), N(i,j), 1.0e-12);
        }
    }

    // Change value of middle element of original matrix
    M(1, 1) = 20.0;
    // Check corresponding value in copied matrix has been udpated
    ASSERT_NEAR(N(1, 1), 20.0, 1.0e-12);

    // Change value of last element in copied matrix
    N(2, 2) = 50.0;
    // Check corresponding value in original matrix has been udpated
    ASSERT_NEAR(M(2, 2), 50.0, 1.0e-12);
}

TEST(MatrixTest, DeepCopyConstructor) {
    // Make a vector of values 
    std::vector<double> values = isce::core::arange(0.0, 9.0, 1.0);
    // Make a const matrix from the vector
    const isce::core::Matrix<double> M(values, 3);
    // Make a deep copy (by passing in const matrix)
    isce::core::Matrix<double> N(M);

    // Change value of middle element of copied matrix
    N(1, 1) = 20.0;
    // Check corresponding value in original matrix has NOT been udpated
    ASSERT_NEAR(M(1, 1), 4.0, 1.0e-12);
}

TEST(MatrixTest, MatrixView) {
    // Make a vector of values 
    std::vector<double> values = isce::core::arange(0.0, 9.0, 1.0);
    // Make a matrix from the vector
    isce::core::Matrix<double> M(values, 3);
    // Get a view of a subset of the matrix
    const isce::core::Matrix<double>::view_t view = M.submat(1, 1, 2, 2);

    // Vector of expected values
    std::vector<double> expected{4.0, 5.0, 7.0, 8.0};

    // Compare values
    size_t count = 0;
    for (auto it = view.begin(); it != view.end(); ++it) {
        double view_val = *it;
        ASSERT_NEAR(view_val, expected[count], 1.0e-12);
        ++count;
    }
}

TEST(MatrixTest, MatrixViewConstructor) {
    // Make a vector of values 
    std::vector<double> values = isce::core::arange(0.0, 9.0, 1.0);
    // Make a matrix from the vector
    isce::core::Matrix<double> M(values, 3);
    // Construct a new matrix from a view of a subset of the matrix
    isce::core::Matrix<double>N(M.submat(1, 1, 2, 2));

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
    std::vector<double> values = isce::core::arange(0.0, 9.0, 1.0);
    // Make a matrix from the vector
    isce::core::Matrix<double> M(values, 3);

    // Make a new matrix of zeros
    isce::core::Matrix<double> N(3, 3);
    N.zeros();

    // Set column of matrix with row from original matrix
    N.submat(0, 1, 3, 1) = M.submat(1, 0, 1, 3);

    // Vector of expected values
    std::vector<double> expected{0.0, 3.0, 0.0,
                                 0.0, 4.0, 0.0,
                                 0.0, 5.0, 0.0};

    // Compare values
    for (size_t count = 0; count < (N.width() * N.length()); ++count) {
        ASSERT_NEAR(N(count), expected[count], 1.0e-12);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
