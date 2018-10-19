//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018
//

#include <cmath>
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"

// isce::core
#include "isce/core/Constants.h"
#include "isce/core/Interpolator.h"
using isce::core::Matrix;

void loadInterpData(Matrix<double> &);

// Function to return a Python style arange vector
std::vector<double> arange(double low, double high, double increment) {
    // Instantiate the vector
    std::vector<double> data;
    // Set the first value
    double current = low;
    // Loop over the increments and add to vector
    while (current < high) {
        data.push_back(current);
        current += increment;
    }
    // done
    return data;
}

struct InterpolatorTest : public ::testing::Test {

    // The low resolution data
    isce::core::Matrix<double> M;
    // The low resolution indices
    std::vector<double> xindex;
    std::vector<double> yindex;
    // Truth data
    isce::core::Matrix<double> true_values;

    double start, delta;

    protected:
        // Constructor
        InterpolatorTest() {

            // Create indices
            xindex = arange(-5.01, 5.01, 0.25);
            yindex = arange(-5.01, 5.01, 0.25);
            size_t nx = xindex.size();
            size_t ny = yindex.size();

            // Allocate the matrix
            M.resize(ny, nx);

            // Fill matrix values with function z = sin(x**2 + y**2)
            for (size_t i = 0; i < ny; ++i) {
                for (size_t j = 0; j < nx; ++j) {
                    M(i,j) = std::sin(yindex[i]*yindex[i] + xindex[j]*xindex[j]);
                }
            }

            // Read the truth data
            loadInterpData(true_values);

            // Starting coordinate and spacing of data
            start = -5.01;
            delta = 0.25;
        }
};

// Test bilinear interpolation
TEST_F(InterpolatorTest, Bilinear) {
    size_t N_pts = true_values.length();
    double error = 0.0;
    // Create interpolator
    isce::core::BilinearInterpolator<double> interp;
    // Loop over test points
    for (size_t i = 0; i < N_pts; ++i) {
        // Unpack location to interpolate
        const double x = (true_values(i,0) - start) / delta;
        const double y = (true_values(i,1) - start) / delta;
        const double zref = true_values(i,2);
        // Perform interpolation
        double z = interp.interpolate(x, y, M);
        // Check
        ASSERT_NEAR(z, zref, 1.0e-8);
        // Accumulate error
        error += std::pow(z - true_values(i,5), 2);
    }
    ASSERT_TRUE((error / N_pts) < 0.07);
}

// Test bicubic interpolation
// Simply test final sum of square errors
TEST_F(InterpolatorTest, Bicubic) {
    size_t N_pts = true_values.length();
    double error = 0.0;
    // Create interpolator
    isce::core::BicubicInterpolator<double> interp;
    // Loop over test points
    for (size_t i = 0; i < N_pts; ++i) {
        // Unpack location to interpolate
        const double x = (true_values(i,0) - start) / delta;
        const double y = (true_values(i,1) - start) / delta;
        const double zref = true_values(i,5);
        // Perform interpolation
        double z = interp.interpolate(x, y, M);
        // Accumulate error
        error += std::pow(z - zref, 2);
    }
    ASSERT_TRUE((error / N_pts) < 0.058);
}

// Test biquintic spline interpolation
TEST_F(InterpolatorTest, Biquintic) {
    size_t N_pts = true_values.length();
    double error = 0.0;
    // Create interpolator
    isce::core::Spline2dInterpolator<double> interp(6);
    // Loop over test points
    for (size_t i = 0; i < N_pts; ++i) {
        // Unpack location to interpolate
        const double x = (true_values(i,0) - start) / delta;
        const double y = (true_values(i,1) - start) / delta;
        const double zref = true_values(i,5);
        // Perform interpolation
        double z = interp.interpolate(x, y, M);
        // Accumulate error
        error += std::pow(z - zref, 2);
    }
    ASSERT_TRUE((error / N_pts) < 0.058);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

void loadInterpData(Matrix<double> & M) {
    /*
    Load ground truth interpolation data. The test data is the function:

    z = sqrt(x^2 + y^2)

    The columns of the data are:
    x_index    y_index    bilinear_interp    bicubic_interp    5thorder_spline    truth
    */

    // Open file for reading
    std::ifstream fid("data.txt");
    // Check if file open was successful
    if (fid.fail()) {
        std::cout << "Error: Failed to open data file for interpolator test." << std::endl;
    }

    std::vector<double> xvec, yvec, zlinear_vec, zcubic_vec, zquintic_vec, ztrue_vec;

    // Loop over interpolation data
    while (fid) {

        // Parse line
        std::string str;
        std::stringstream stream;
        double x, y, z_linear, z_cubic, z_quintic, z_true;

        std::getline(fid, str);
        if (str.length() < 1)
            break;
        stream << str;
        stream >> x >> y >> z_linear >> z_cubic >> z_quintic >> z_true;

        // Add data to orbit
        xvec.push_back(x);
        yvec.push_back(y);
        zlinear_vec.push_back(z_linear);
        zcubic_vec.push_back(z_cubic);
        zquintic_vec.push_back(z_quintic);
        ztrue_vec.push_back(z_true);
    }

    // Close the file
    fid.close();

    // Fill the matrix
    const size_t N = xvec.size();
    M.resize(N, 6);
    for (size_t i = 0; i < N; ++i) {
        M(i,0) = xvec[i];
        M(i,1) = yvec[i];
        M(i,2) = zlinear_vec[i];
        M(i,3) = zcubic_vec[i];
        M(i,4) = zquintic_vec[i];
        M(i,5) = ztrue_vec[i];
    }
}

// end of file
