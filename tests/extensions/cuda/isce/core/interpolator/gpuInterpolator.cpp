//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright 2018
//

#include <cmath>
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"

#include "isce/core/Constants.h"
#include "isce/core/Interpolator.h"
#include "isce/cuda/core/gpuInterpolator.h"

using isce::core::Matrix;
using isce::core::Matrix;
using isce::cuda::core::gpuInterpolator;
using isce::cuda::core::gpuBilinearInterpolator;
using isce::cuda::core::gpuBicubicInterpolator;
using isce::cuda::core::gpuSpline2dInterpolator;

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

struct gpuInterpolatorTest : public ::testing::Test {

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
        gpuInterpolatorTest() {

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
TEST_F(gpuInterpolatorTest, BilinearDouble) {
    size_t N_pts = true_values.length();
    double error = 0.0;
    std::vector<double> v_z(true_values.length());
    double *z = v_z.data();
    
    // instantiate parent and derived class
    gpuBilinearInterpolator<double> gpu_bilinear;

    // Perform interpolation
    gpu_bilinear.interpolate_h(true_values, M, start, delta, z);

    for (size_t i = 0; i < N_pts; ++i) {
        // Unpack reference value
        const double zref = true_values(i,2);
        // Check
        ASSERT_NEAR(z[i], zref, 1.0e-8);
        // Accumulate error
        error += std::pow(z[i] - true_values(i,5), 2);
    }
    ASSERT_TRUE((error / N_pts) < 0.07);
}

// Test bicubic interpolation
TEST_F(gpuInterpolatorTest, BicubicDouble) {
    size_t N_pts = true_values.length();
    double error = 0.0;
    std::vector<double> v_z(true_values.length());
    double *z = v_z.data();
    
    // instantiate parent and derived class
    gpuBicubicInterpolator<double> gpu_bicubic;

    // Perform interpolation
    gpu_bicubic.interpolate_h(true_values, M, start, delta, z);

    for (size_t i = 0; i < N_pts; ++i) {
        // Unpack reference value
        const double zref = true_values(i,5);
        // Accumulate error
        error += std::pow(z[i] - zref, 2);
    }
    ASSERT_TRUE((error / N_pts) < 0.058);
}

// Test bicubic interpolation
TEST_F(gpuInterpolatorTest, Spline2dDouble) {
    size_t N_pts = true_values.length();
    double error = 0.0;
    std::vector<double> v_z(true_values.length());
    double *z = v_z.data();
    
    // instantiate parent and derived class
    gpuSpline2dInterpolator<double> gpu_spline2d(6);

    // Perform interpolation
    gpu_spline2d.interpolate_h(true_values, M, start, delta, z);

    for (size_t i = 0; i < N_pts; ++i) {
        // Unpack reference value
        const double zref = true_values(i,5);
        // Accumulate error
        error += std::pow(z[i] - zref, 2);
    }
    ASSERT_TRUE((error / N_pts) < 0.058);
}

/*
// Test bicubic interpolation
// Simply test final sum of square errors

// Test biquintic spline interpolation
TEST_F(gpuInterpolatorTest, Biquintic) {
    size_t N_pts = true_values.length();
    double error = 0.0;
    for (size_t i = 0; i < N_pts; ++i) {
        // Unpack location to interpolate
        const double x = (true_values(i,0) - start) / delta;
        const double y = (true_values(i,1) - start) / delta;
        const double zref = true_values(i,5);
        // Perform interpolation
        double z = isce::core::Interpolator::interp_2d_spline(6, M, x, y);
        // Accumulate error
        error += std::pow(z - zref, 2);
    }
    ASSERT_TRUE((error / N_pts) < 0.058);
}
*/


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
