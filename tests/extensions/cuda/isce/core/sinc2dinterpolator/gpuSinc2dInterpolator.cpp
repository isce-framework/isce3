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
#include <complex>
#include "gtest/gtest.h"

#include "isce/core/Constants.h"
#include "isce/core/Interpolator.h"
#include "isce/cuda/core/gpuInterpolator.h"
#include "isce/cuda/core/gpuComplex.h"

using isce::core::Matrix;
using isce::core::Sinc2dInterpolator;
using isce::cuda::core::gpuComplex;
using isce::cuda::core::gpuInterpolator;
using isce::cuda::core::gpuSinc2dInterpolator;

void loadChipData(Matrix<std::complex<float>> &, size_t, size_t);

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

struct gpuSinc2dInterpolatorTest : public ::testing::Test {

    // The low resolution data
    Matrix<std::complex<float>> chip;
    // The low resolution indices
    float xindex;
    float yindex;
    // Truth data

    double start, delta;

    protected:
        // Constructor
        gpuSinc2dInterpolatorTest() {

            // Create indices
            xindex = 5.623924;
            yindex = 5.430916;
            size_t nx = 9;
            size_t ny = 9;

            // Allocate the matrix
            chip.resize(ny, nx);

            // Read the truth data
            loadChipData(chip, nx, ny);
        }
};

// Test sinc2d interpolation
TEST_F(gpuSinc2dInterpolatorTest, Sinc2dFloat) {
    isce::core::Matrix<double> indices;
    indices.resize(2,2);
    for (int i = 0; i < 2; ++i) {
        indices(i,0) = xindex;
        indices(i,1) = yindex;
    }

    size_t N_pts = indices.length();
    double error = 0.0;
    gpuComplex<float> gpu_z[2];
    std::complex<float> cpu_z;
    
    // instantiate GPU and CPU class
    gpuSinc2dInterpolator<gpuComplex<float>> gpuSinc2d(
                  isce::core::SINC_LEN, isce::core::SINC_SUB);
    Sinc2dInterpolator<std::complex<float>> cpuSinc2d(
                  isce::core::SINC_LEN, isce::core::SINC_SUB);

    // Perform interpolation
    isce::core::Matrix<gpuComplex<float>> gpu_chip(chip.length(), chip.width());
    for (int i = 0; i < chip.length(); ++i) {
        for (int j = 0; j < chip.width(); ++j)
            gpu_chip(i,j) = gpuComplex<float>(std::real(chip(i,j)), std::imag(chip(i,j)));
    }
        
    cpu_z = cpuSinc2d.interpolate(xindex, yindex, chip);
    printf("cpu_z %f %f\n", std::real(cpu_z), std::imag(cpu_z));
    gpuSinc2d.interpolate_h(indices, gpu_chip, start, delta, gpu_z);

    for (int i=0; i<2; ++i) {
        printf("gpu_z %f %f\n", gpu_z[i].r, gpu_z[i].i);
        ASSERT_NEAR(gpu_z[i].r, std::real(cpu_z), 1.0e-8);
        ASSERT_NEAR(gpu_z[i].i, std::imag(cpu_z), 1.0e-8);
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

void loadChipData(Matrix<std::complex<float>> & chip, size_t nx, size_t ny) {
    /*
     * pulled chip values from i=100, j=100 in resampslc test
    */

    // Open file for reading
    std::ifstream fid("chip.txt");
    // Check if file open was successful
    if (fid.fail()) {
        std::cout << "Error: Failed to open data file for interpolator test." << std::endl;
    }

    // Loop over interpolation data
    while (fid) {

        // Parse line
        std::string str;
        std::stringstream stream;
        int ii, jj;
        float r, i;

        std::getline(fid, str);
        if (str.length() < 1)
            break;
        stream << str;
        stream >> ii >> jj >> r >> i;

        // Add data to orbit
        chip(ii,jj) = std::complex<float>(r ,i);
    }

    // Close the file
    fid.close();
}

// end of file
