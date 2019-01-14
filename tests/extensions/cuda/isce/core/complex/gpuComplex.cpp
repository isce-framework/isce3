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
    size_t n_data_pts = 500;
    valarray<float> a_float_real = valarray<float>(n_data_pts);
    valarray<float> b_float_real = valarray<float>(n_data_pts);
    valarray<complex<float>> a_float_cpu_complex;
    valarray<complex<float>> b_float_cpu_complex;
    valarray<gpuComplex<float>> a_float_gpu_complex = valarray<gpuComplex<float>>(n_data_pts);
    valarray<gpuComplex<float>> b_float_gpu_complex = valarray<gpuComplex<float>>(n_data_pts);
    valarray<double> a_double_real = valarray<double>(n_data_pts);
    valarray<double> b_double_real = valarray<double>(n_data_pts);
    valarray<complex<double>> a_double_cpu_complex;
    valarray<complex<double>> b_double_cpu_complex;
    valarray<gpuComplex<double>> a_double_gpu_complex = valarray<gpuComplex<double>>(n_data_pts);
    valarray<gpuComplex<double>> b_double_gpu_complex = valarray<gpuComplex<double>>(n_data_pts);
    unsigned fails;

    protected:
        // constructor
        gpuComplexTest() {
            // create all the float test data
            makeRandomReal<float>(a_float_real, n_data_pts);
            makeRandomReal<float>(b_float_real, n_data_pts);
            a_float_cpu_complex = makeRandomStdComplex<float>(n_data_pts);
            b_float_cpu_complex = makeRandomStdComplex<float>(n_data_pts);
            memcpy(&a_float_gpu_complex[0], &a_float_cpu_complex[0], n_data_pts*sizeof(complex<float>));
            memcpy(&b_float_gpu_complex[0], &b_float_cpu_complex[0], n_data_pts*sizeof(complex<float>));
            // create all the double test data
            makeRandomReal<double>(a_double_real, n_data_pts);
            makeRandomReal<double>(b_double_real, n_data_pts);
            a_double_cpu_complex = makeRandomStdComplex<double>(n_data_pts);
            b_double_cpu_complex = makeRandomStdComplex<double>(n_data_pts);
            memcpy(&a_double_gpu_complex[0], &a_double_cpu_complex[0], n_data_pts*sizeof(complex<double>));
            memcpy(&b_double_gpu_complex[0], &b_double_cpu_complex[0], n_data_pts*sizeof(complex<double>));

        }
};

// test then copy, paste, modify 63x for other operator combinations
// add operator float
TEST_F(gpuComplexTest, AddComplexFloatAndComplexFloat) {
    auto c_cpu = a_float_cpu_complex + b_float_cpu_complex;
    auto c_gpu = a_float_gpu_complex + b_float_gpu_complex;
    for (int i=0; i<n_data_pts; ++i) {
        ASSERT_EQ(std::real(c_cpu[i]), c_gpu[i].r);
        ASSERT_EQ(std::imag(c_cpu[i]), c_gpu[i].i);
    }
}

TEST_F(gpuComplexTest, AddComplexFloatAndFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i] + b_float_real[i];
        auto c_gpu = a_float_gpu_complex[i] + b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, AddFloatAndComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = complex<float>(b_float_real[i]+std::real(a_float_cpu_complex[i]), std::imag(a_float_cpu_complex[i]));
        gpuComplex<float> c_gpu = b_float_real[i] + a_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, AddComplexFloatAndDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = complex<float>(std::real(a_float_cpu_complex[i]) + b_double_real[i], std::imag(a_float_cpu_complex[i]));
        auto c_gpu = a_float_gpu_complex[i] + b_double_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, AddDoubleAndComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = complex<float>(b_double_real[i] + std::real(a_float_cpu_complex[i]), std::imag(a_float_cpu_complex[i]));
        gpuComplex<float> c_gpu = b_double_real[i] + a_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

// subtract operator float
TEST_F(gpuComplexTest, SubtractComplexFloatAndComplexFloat) {
    auto c_cpu = a_float_cpu_complex - b_float_cpu_complex;
    auto c_gpu = a_float_gpu_complex - b_float_gpu_complex;
    for (int i=0; i<n_data_pts; ++i) {
        ASSERT_EQ(std::real(c_cpu[i]), c_gpu[i].r);
        ASSERT_EQ(std::imag(c_cpu[i]), c_gpu[i].i);
    }
}

TEST_F(gpuComplexTest, SubtractComplexFloatAndFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i] - b_float_real[i];
        auto c_gpu = a_float_gpu_complex[i] - b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, SubtractFloatAndComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = complex<float>(b_float_real[i] - std::real(a_float_cpu_complex[i]), -std::imag(a_float_cpu_complex[i]));
        gpuComplex<float> c_gpu = b_float_real[i] - a_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, SubtractComplexFloatAndDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = complex<float>(std::real(a_float_cpu_complex[i]) - b_double_real[i], std::imag(a_float_cpu_complex[i]));
        auto c_gpu = a_float_gpu_complex[i] - b_double_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, SubtractDoubleAndComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = complex<float>(b_double_real[i] - std::real(a_float_cpu_complex[i]), -std::imag(a_float_cpu_complex[i]));
        gpuComplex<float> c_gpu = b_double_real[i] - a_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

// multiply operator float
TEST_F(gpuComplexTest, MultiplyComplexFloatAndComplexFloat) {
    auto c_cpu = a_float_cpu_complex * b_float_cpu_complex;
    auto c_gpu = a_float_gpu_complex * b_float_gpu_complex;
    for (int i=0; i<n_data_pts; ++i) {
        ASSERT_EQ(std::real(c_cpu[i]), c_gpu[i].r);
        ASSERT_EQ(std::imag(c_cpu[i]), c_gpu[i].i);
    }
}

TEST_F(gpuComplexTest, MultiplyComplexFloatAndFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i] * b_float_real[i];
        auto c_gpu = a_float_gpu_complex[i] * b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, MultiplyFloatAndComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = b_float_real[i] * a_float_cpu_complex[i];
        gpuComplex<float> c_gpu = b_float_real[i] * a_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, MultiplyComplexFloatAndDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = complex<float>(std::real(a_float_cpu_complex[i]) * b_double_real[i], std::imag(a_float_cpu_complex[i]) * b_double_real[i]);
        auto c_gpu = a_float_gpu_complex[i] * b_double_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, MultiplyDoubleAndComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = complex<float>(b_double_real[i] * std::real(a_float_cpu_complex[i]), b_double_real[i] * std::imag(a_float_cpu_complex[i]));
        gpuComplex<float> c_gpu = b_double_real[i] * a_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

// divide operator float
TEST_F(gpuComplexTest, DivideComplexFloatAndComplexFloat) {
    auto c_cpu = a_float_cpu_complex / b_float_cpu_complex;
    auto c_gpu = a_float_gpu_complex / b_float_gpu_complex;
    for (int i=0; i<n_data_pts; ++i) {
        ASSERT_LT(abs(std::real(c_cpu[i]) - c_gpu[i].r), 1e-6);
        ASSERT_LT(abs(std::imag(c_cpu[i]) - c_gpu[i].i), 1e-6);
    }
}

TEST_F(gpuComplexTest, DivideComplexFloatAndFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i] / b_float_real[i];
        auto c_gpu = a_float_gpu_complex[i] / b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, DivideFloatAndComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = b_float_real[i] / a_float_cpu_complex[i];
        gpuComplex<float> c_gpu = b_float_real[i] / a_float_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-5);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-5);
    }
}

TEST_F(gpuComplexTest, DivideComplexFloatAndDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = complex<float>(std::real(a_float_cpu_complex[i]) / b_double_real[i], std::imag(a_float_cpu_complex[i]) / b_double_real[i]);
        auto c_gpu = a_float_gpu_complex[i] / b_double_real[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, DivideDoubleAndComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<float> c_cpu = (float)b_double_real[i] / a_float_cpu_complex[i];
        gpuComplex<float> c_gpu = b_double_real[i] / a_float_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

// add operator double
TEST_F(gpuComplexTest, AddComplexDoubleAndComplexDouble) {
    auto c_cpu = a_double_cpu_complex + b_double_cpu_complex;
    auto c_gpu = a_double_gpu_complex + b_double_gpu_complex;
    for (int i=0; i<n_data_pts; ++i) {
        ASSERT_EQ(std::real(c_cpu[i]), c_gpu[i].r);
        ASSERT_EQ(std::imag(c_cpu[i]), c_gpu[i].i);
    }
}

TEST_F(gpuComplexTest, AddComplexDoubleAndFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<double> c_cpu = complex<double>(std::real(a_double_cpu_complex[i]) + b_float_real[i], std::imag(a_double_cpu_complex[i]));
        auto c_gpu = a_double_gpu_complex[i] + b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(a_double_cpu_complex[i]), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, AddFloatAndComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        gpuComplex<double> c_gpu = b_float_real[i] + a_double_gpu_complex[i];
        ASSERT_EQ(b_float_real[i]+std::real(a_double_cpu_complex[i]), c_gpu.r);
        ASSERT_EQ(std::imag(a_double_cpu_complex[i]), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, AddComplexDoubleAndDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i] + b_double_real[i];
        auto c_gpu = a_double_gpu_complex[i] + b_double_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, AddDoubleAndComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<double> c_cpu = a_double_cpu_complex[i] + b_double_real[i];
        gpuComplex<double> c_gpu = a_double_gpu_complex[i] + b_double_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

// subtract operator double
TEST_F(gpuComplexTest, SubtractComplexDoubleAndComplexDouble) {
    auto c_cpu = a_double_cpu_complex - b_double_cpu_complex;
    auto c_gpu = a_double_gpu_complex - b_double_gpu_complex;
    for (int i=0; i<n_data_pts; ++i) {
        ASSERT_EQ(std::real(c_cpu[i]), c_gpu[i].r);
        ASSERT_EQ(std::imag(c_cpu[i]), c_gpu[i].i);
    }
}

TEST_F(gpuComplexTest, SubtractComplexDoubleAndFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_gpu = a_double_gpu_complex[i] - b_float_real[i];
        ASSERT_EQ(std::real(a_double_cpu_complex[i]) - b_float_real[i], c_gpu.r);
        ASSERT_EQ(std::imag(a_double_cpu_complex[i]), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, SubtractFloatAndComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        gpuComplex<double> c_gpu = b_float_real[i] - a_double_gpu_complex[i];
        ASSERT_EQ(b_float_real[i] - std::real(a_double_cpu_complex[i]), c_gpu.r);
        ASSERT_EQ(std::imag(-a_double_cpu_complex[i]), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, SubtractComplexDoubleAndDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i] - b_double_real[i];
        auto c_gpu = a_double_gpu_complex[i] - b_double_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, SubtractDoubleAndComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<double> c_cpu = a_double_cpu_complex[i] - b_double_real[i];
        gpuComplex<double> c_gpu = a_double_gpu_complex[i] - b_double_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

// multiply operator double
TEST_F(gpuComplexTest, MultiplyComplexDoubleAndComplexDouble) {
    auto c_cpu = a_double_cpu_complex * b_double_cpu_complex;
    auto c_gpu = a_double_gpu_complex * b_double_gpu_complex;
    for (int i=0; i<n_data_pts; ++i) {
        ASSERT_EQ(std::real(c_cpu[i]), c_gpu[i].r);
        ASSERT_EQ(std::imag(c_cpu[i]), c_gpu[i].i);
    }
}

TEST_F(gpuComplexTest, MultiplyComplexDoubleAndFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<double> c_cpu = complex<double>(std::real(a_double_cpu_complex[i]) * b_float_real[i], std::imag(a_double_cpu_complex[i]) * b_float_real[i]);
        auto c_gpu = a_double_gpu_complex[i] * b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, MultiplyFloatAndComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<double> c_cpu = complex<double>(b_float_real[i] * std::real(a_double_cpu_complex[i]), b_float_real[i] * std::imag(a_double_cpu_complex[i]));
        gpuComplex<double> c_gpu = b_float_real[i] * a_double_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, MultiplyComplexDoubleAndDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i] * b_double_real[i];
        auto c_gpu = a_double_gpu_complex[i] * b_double_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, MultiplyDoubleAndComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<double> c_cpu = b_double_real[i] * a_double_cpu_complex[i];
        gpuComplex<double> c_gpu = b_double_real[i] * a_double_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

// divide operator double
TEST_F(gpuComplexTest, DivideComplexDoubleAndComplexDouble) {
    auto c_cpu = a_double_cpu_complex / b_double_cpu_complex;
    auto c_gpu = a_double_gpu_complex / b_double_gpu_complex;
    for (int i=0; i<n_data_pts; ++i) {
        ASSERT_LT(abs(std::real(c_cpu[i]) - c_gpu[i].r), 1e-6);
        ASSERT_LT(abs(std::imag(c_cpu[i]) - c_gpu[i].i), 1e-6);
    }
}

TEST_F(gpuComplexTest, DivideComplexDoubleAndFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i] / double(b_float_real[i]);
        auto c_gpu = a_double_gpu_complex[i] / b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, DivideFloatAndComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<double> c_cpu = double(b_float_real[i]) / a_double_cpu_complex[i];
        gpuComplex<double> c_gpu = b_float_real[i] / a_double_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-5);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-5);
    }
}

TEST_F(gpuComplexTest, DivideComplexDoubleAndDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<double> c_cpu = a_double_cpu_complex[i] / b_double_real[i];
        auto c_gpu = a_double_gpu_complex[i] / b_double_real[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, DivideDoubleAndComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        complex<double> c_cpu = b_double_real[i] / a_double_cpu_complex[i];
        gpuComplex<double> c_gpu = b_double_real[i] / a_double_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

// compound assignment operators complex<double> and float
TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleAddFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu += b_float_real[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu += b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleSubtractFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu -= b_float_real[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu -= b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleMultiplyFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu *= b_float_real[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu *= b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleDivideFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu /= b_float_real[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu /= b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

// compound assignment operators complex<double> and double
TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleAddDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu += b_double_real[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu += b_double_real[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleSubtractDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu -= b_double_real[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu -= b_double_real[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleMultiplyDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu *= b_double_real[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu *= b_double_real[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-3);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-3);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleDivideDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu /= b_double_real[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu /= b_double_real[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

// compound assignment operators complex<double> and complex<float>
TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleAddComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu += b_float_cpu_complex[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu += b_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleSubtractComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu -= b_float_cpu_complex[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu -= b_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleMultiplyComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu *= b_float_cpu_complex[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu *= b_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleDivideComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu /= b_float_cpu_complex[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu /= b_float_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

// compound assignment operators complex<double> and complex<double>
TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleAddComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu += b_double_cpu_complex[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu += b_double_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleSubtractComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu -= b_double_cpu_complex[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu -= b_double_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleMultiplyComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu *= b_double_cpu_complex[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu *= b_double_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-3);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-3);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexDoubleDivideComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_double_cpu_complex[i];
        c_cpu /= b_double_cpu_complex[i];
        auto c_gpu = a_double_gpu_complex[i];
        c_gpu /= b_double_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

// compound assignment operators complex<float> and float
TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatAddFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu += b_float_real[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu += b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatSubtractFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu -= b_float_real[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu -= b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatMultiplyFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu *= b_float_real[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu *= b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatDivideFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu /= b_float_real[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu /= b_float_real[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

// compound assignment operators complex<float> and double
TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatAddDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu += b_double_real[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu += b_double_real[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatSubtractDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu -= b_double_real[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu -= b_double_real[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatMultiplyDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu *= b_double_real[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu *= b_double_real[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-3);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-3);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatDivideDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu /= b_double_real[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu /= b_double_real[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

// compound assignment operators complex<float> and complex<float>
TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatAddComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu += b_float_cpu_complex[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu += b_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatSubtractComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu -= b_float_cpu_complex[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu -= b_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatMultiplyComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu *= b_float_cpu_complex[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu *= b_float_gpu_complex[i];
        ASSERT_EQ(std::real(c_cpu), c_gpu.r);
        ASSERT_EQ(std::imag(c_cpu), c_gpu.i);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatDivideComplexFloat) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu /= b_float_cpu_complex[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu /= b_float_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

// compound assignment operators complex<float> and complex<double>
TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatAddComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu += b_double_cpu_complex[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu += b_double_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatSubtractComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu -= b_double_cpu_complex[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu -= b_double_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatMultiplyComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu *= b_double_cpu_complex[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu *= b_double_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-3);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-3);
    }
}

TEST_F(gpuComplexTest, CompoundAssignmentComplexFloatDivideComplexDouble) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = a_float_cpu_complex[i];
        c_cpu /= b_double_cpu_complex[i];
        auto c_gpu = a_float_gpu_complex[i];
        c_gpu /= b_double_gpu_complex[i];
        ASSERT_LT(abs(std::real(c_cpu) - c_gpu.r), 1e-4);
        ASSERT_LT(abs(std::imag(c_cpu) - c_gpu.i), 1e-4);
    }
}

TEST_F(gpuComplexTest, DoubleAbs) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = std::abs(a_double_cpu_complex[i]);
        auto c_gpu = abs(a_double_gpu_complex[i]);
        ASSERT_LT(abs(c_cpu - c_gpu), 1e-4);
        ASSERT_LT(abs(c_cpu - c_gpu), 1e-4);
    }
}

TEST_F(gpuComplexTest, FloatAbs) {
    for (int i=0; i<n_data_pts; ++i) {
        auto c_cpu = std::abs(a_float_cpu_complex[i]);
        auto c_gpu = abs(a_float_gpu_complex[i]);
        ASSERT_LT(abs(c_cpu - c_gpu), 1e-4);
        ASSERT_LT(abs(c_cpu - c_gpu), 1e-4);
    }
}

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
