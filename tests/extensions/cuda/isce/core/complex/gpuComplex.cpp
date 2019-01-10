// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright 2018
//

#include <cmath>
#include <iostream>
#include <valarray>
#include "isce/cuda/core/gpuComplex.h"
#include "gtest/gtest.h"
#include "gpuComplex.h"

using isce::cuda::core::gpuComplex;
using std::endl;
using std::valarray;
using std::complex;

struct gpuComplexTest : public ::testing::Test {
    virtual void SetUp() {
        fails = 0;
    }
    virtual void TearDown() {
        if (fails > 0) {
            std::cerr << "gpuComplex::TearDown sees failures" << std::endl;
        }
    }
    size_t n_data_pts = 10000;
    valarray<float> a_float_real = valarray<float>(n_data_pts);
    valarray<float> b_float_real = valarray<float>(n_data_pts);
    valarray<complex<float>> a_float_cpu_complex = valarray<complex<float>>(n_data_pts);
    valarray<complex<float>> b_float_cpu_complex = valarray<complex<float>>(n_data_pts);
    valarray<gpuComplex<float>> a_float_gpu_complex = valarray<gpuComplex<float>>(n_data_pts);
    valarray<gpuComplex<float>> b_float_gpu_complex = valarray<gpuComplex<float>>(n_data_pts);
    valarray<double> a_double_real = valarray<double>(n_data_pts);
    valarray<double> b_double_real = valarray<double>(n_data_pts);
    valarray<complex<double>> a_double_cpu_complex = valarray<complex<double>>(n_data_pts);
    valarray<complex<double>> b_double_cpu_complex = valarray<complex<double>>(n_data_pts);
    valarray<gpuComplex<double>> a_double_gpu_complex = valarray<gpuComplex<double>>(n_data_pts);
    valarray<gpuComplex<double>> b_double_gpu_complex = valarray<gpuComplex<double>>(n_data_pts);
    unsigned fails;

    protected:
        // constructor
        gpuComplexTest() {
            // create all the float test data
            makeRandomReal<float>(a_float_real, n_data_pts);
            makeRandomReal<float>(b_float_real, n_data_pts);
            makeRandomStdComplex<complex<float>>(a_float_cpu_complex, n_data_pts);
            makeRandomStdComplex<complex<float>>(b_float_cpu_complex, n_data_pts);
            memcpy(&a_float_gpu_complex[0], &a_float_cpu_complex[0], n_data_pts*sizeof(complex<float>));
            memcpy(&b_float_gpu_complex[0], &b_float_cpu_complex[0], n_data_pts*sizeof(complex<float>));
            // create all the double test data
            makeRandomReal<double>(a_double_real, n_data_pts);
            makeRandomReal<double>(b_double_real, n_data_pts);
            makeRandomStdComplex<complex<double>>(a_double_cpu_complex, n_data_pts);
            makeRandomStdComplex<complex<double>>(b_double_cpu_complex, n_data_pts);
            memcpy(&a_double_gpu_complex[0], &a_double_cpu_complex[0], n_data_pts*sizeof(complex<double>));
            memcpy(&b_double_gpu_complex[0], &b_double_cpu_complex[0], n_data_pts*sizeof(complex<double>));

        }
};

// test then copy, paste, modify 63x for other operator combinations
TEST_F(gpuComplexTest, AddComplexFloatComplexFloat) {
    auto c_cpu = a_float_cpu_complex + b_float_cpu_complex;
    auto c_gpu = a_float_gpu_complex + b_float_gpu_complex;

    for (int i=0; i<n_data_pts; ++i) {
        ASSERT_NEAR(std::real(c_cpu[i]), c_gpu[i].r, 1.0e-6);
        ASSERT_NEAR(std::imag(c_cpu[i]), c_gpu[i].i, 1.0e-6);
    }

    fails += ::testing::Test::HasFailure();
}


int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
