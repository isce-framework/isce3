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
#include "isce/core/Cube.h"

TEST(CubeTest, SimpleConstructor) {
    // Make a cube with a fixed shape
    isce::core::Cube<double> M(3, 4, 5);
    ASSERT_EQ(M.height(), 3);
    ASSERT_EQ(M.length(), 4);
    ASSERT_EQ(M.width(), 5);
}

TEST(CubeTest, Resize) {
    // Make a cube with a fixed shape
    isce::core::Cube<double> M(3, 4, 5);
    // Resize it
    M.resize(2, 3, 4);
    // Check shape 
    ASSERT_EQ(M.height(), 2);
    ASSERT_EQ(M.length(), 3);
    ASSERT_EQ(M.width(), 4);
}

TEST(CubeTest, FixedValues) {
    // Make a cube with a fixed shape
    isce::core::Cube<double> M(3, 4, 5);

    // Fill it with zeros and check values
    M.zeros();
    for (size_t count = 0; count < (M.height() * M.width() * M.length()); ++count) {
        ASSERT_NEAR(M(count), 0.0, 1.0e-12);
    }

    // Fill it with a constant value and check
    M.fill(10.0);
    for (size_t count = 0; count < (M.height() * M.width() * M.length()); ++count) {
        ASSERT_NEAR(M(count), 10.0, 1.0e-12);
    }
}

TEST(CubeTest, VectorConstructor) {
    // Make a vector of values
    std::vector<double> values = isce::core::arange(0.0, 60.0, 1.0);
    // Make a cube from the vector
    isce::core::Cube<double> M(values, 4, 5);

    // Check the shape
    ASSERT_EQ(M.height(), 3);
    ASSERT_EQ(M.length(), 4);
    ASSERT_EQ(M.width(), 5);

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

TEST(CubeTest, CopyConstructor) {
    // Make a vector of values 
    std::vector<double> values = isce::core::arange(0.0, 60.0, 1.0);
    // Make a cube from the vector
    isce::core::Cube<double> M(values, 4, 5);
    // Make a shallow copy
    isce::core::Cube<double> N(M);

    // Check shapes are equal
    ASSERT_EQ(M.height(), N.height());
    ASSERT_EQ(M.width(), N.width());
    ASSERT_EQ(M.length(), N.length());

    // Check the values
    for (size_t h=0; h < M.height(); ++h) {
        for (size_t i = 0; i < M.length(); ++i) {
            for (size_t j = 0; j < M.width(); ++j) {
                ASSERT_NEAR(M(h,i,j), N(h,i,j), 1.0e-12);
            }
        }
    }

    // Change value of middle element of original matrix
    M(1, 1, 1) = 20.0;
    // Check corresponding value in copied matrix has been udpated
    ASSERT_NEAR(N(1, 1, 1), 20.0, 1.0e-12);

    // Change value of last element in copied matrix
    N(2, 3, 4) = 50.0;
    // Check corresponding value in original matrix has been udpated
    ASSERT_NEAR(M(2, 3, 4), 50.0, 1.0e-12);
}

TEST(CubeTest, DeepCopyConstructor) {
    // Make a vector of values 
    std::vector<double> values = isce::core::arange(0.0, 60.0, 1.0);
    // Make a const cube from the vector
    const isce::core::Cube<double> M(values, 4, 5);

    //Copy original value in 1,1,1
    double origval = M(1,1,1);

    // Make a deep copy (by passing in const matrix)
    isce::core::Cube<double> N(M);

    // Change value of middle element of copied matrix
    N(1, 1, 1) = 20.0;
    
    // Check corresponding value in original matrix has NOT been udpated
    ASSERT_NEAR(M(1, 1, 1), origval, 1.0e-12);
}

TEST(CubeTest, CubeView) {
    // Make a vector of values 
    std::vector<double> values = isce::core::arange(0.0, 60.0, 1.0);
    // Make a cube from the vector
    isce::core::Cube<double> M(values, 4, 5);
    // Get a view of a subset of the cube
    const isce::core::Cube<double>::view_t view = M.subcube(1, 1, 1, 2, 2, 2);

    // Vector of expected values
    std::vector<double> expected{26.0, 27.0, 31.0, 32.0, 46.0, 47.0, 51.0, 52.0};

    // Compare values
    size_t count = 0;
    for (auto it = view.begin(); it != view.end(); ++it) {
        double view_val = *it;
        ASSERT_NEAR(view_val, expected[count], 1.0e-12);
        ++count;
    }
}

TEST(CubeTest, squareBracket) {
    //Make a vector of values
    std::vector<double> values = isce::core::arange(0.0, 60.0, 1.0);
    //Make a cube from the vector
    isce::core::Cube<double> M(values, 4, 5);
    //Test flat indices
    for (size_t ii=0; ii<60; ii++)
    {
        decltype(M)::index_t ind = { ii/20, (ii%20)/5, ii%5};
        ASSERT_NEAR( M[ind], 1.0*ii, 1.0e-12);
    }
    //Negate all values
    for (size_t ii=0; ii<60; ii++)
    {
        decltype(M)::index_t ind = { ii/20, (ii%20)/5, ii%5};
        M[ind] = -M[ind];
    }
    //Check for negative values
    for (size_t ii=0; ii<60; ii++)
    {
        decltype(M)::index_t ind = { ii/20, (ii%20)/5, ii%5};
        ASSERT_NEAR( M[ind], -1.0*ii, 1.0e-12);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
