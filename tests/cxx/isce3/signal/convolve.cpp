#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include <isce3/signal/convolve.h>
#include <isce3/signal/filterKernel.h>
#include <isce3/signal/multilook.h>

template<class T>
void printValarray(const std::valarray<T>& va, int num)
{
    for (int i = 0; i < va.size() / num; i++) {
        for (int j = 0; j < num; j++) {
            std::cout << va[i * num + j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

double create_data(int line, int col) { return line + col; }

double inner_product(std::valarray<double> x, std::valarray<double> y)
{
    x *= y;
    return x.sum();
}

TEST(Convolve, ConvolveBoxcarKernel)
{

    int length = 200;
    int width = 310;

    int kernel_width = 3;
    int kernel_length = 3;

    // number of pixels padded
    int pad_cols = kernel_width - 1;
    int pad_rows = kernel_length - 1;

    // width and length of the padded buffers
    int width_padded = width + pad_cols;
    int length_padded = length + pad_rows;

    // buffer for data after convolution
    std::valarray<double> filtered_data(length * width);

    // buffer for padded data and mask
    std::valarray<double> data((length_padded) * (width_padded));
    std::valarray<double> mask((length_padded) * (width_padded));

    // mask is one everywhere
    mask = 1.0;

    // create data. Note paddeed boundary is zero
    for (int line = pad_rows / 2; line < pad_rows / 2 + length; ++line) {
        for (int col = pad_cols / 2; col < pad_cols / 2 + width; ++col) {

            data[line * width_padded + col] = create_data(line, col);
        }
    }

    // create the kernels
    std::valarray<double> kernelColumns = isce3::signal::boxcar1D(kernel_width);

    std::valarray<double> kernelRows = isce3::signal::boxcar1D(kernel_length);

    std::valarray<double> kernel2D =
            isce3::signal::boxcar2D(kernel_width, kernel_length);

    // two 1D Convolution in time domain
    isce3::signal::convolve2D(filtered_data, data, mask, kernelColumns,
                              kernelRows, width, width_padded);

    // error
    double max_err = 0.0;

    // buffer for data when kernel is centered on the pixel of interest
    std::valarray<double> d(kernel_width * kernel_length);
    for (int line = pad_rows / 2; line < pad_rows / 2 + length; ++line) {
        for (int col = pad_cols / 2; col < pad_cols / 2 + width; ++col) {
            int kk = 0;
            for (int ii = -kernel_length / 2; ii < kernel_length / 2 + 1;
                 ++ii) {
                for (int jj = -kernel_width / 2; jj < kernel_width / 2 + 1;
                     ++jj) {
                    d[kk] = data[(line + ii) * (width + pad_cols) + col + jj];
                    kk += 1;
                }
            }

            // multiply kernel by data
            double result = inner_product(d, kernel2D);

            // compute the difference
            double diff = std::abs(filtered_data[(line - pad_rows / 2) * width +
                                                 col - pad_cols / 2] -
                                   result);
            if (diff > max_err)
                max_err = diff;
        }
    }

    ASSERT_LT(max_err, 1.0e-12);
}

TEST(Convolve, ConvolveDecimate)
{

    int length = 200;
    int width = 310;

    int kernel_width = 3;
    int kernel_length = 3;

    // number of pixels padded
    int pad_cols = kernel_width - 1;
    int pad_rows = kernel_length - 1;

    // width and length of the padded buffers
    int width_padded = width + pad_cols;
    int length_padded = length + pad_rows;

    // buffer for data after convolution
    int length_decimated = length / kernel_length;
    int width_decimated = width / kernel_width;

    isce3::core::EArray2D<double> filtered_data(length_decimated,
                                                width_decimated);

    // buffer for padded data and mask
    isce3::core::EArray2D<double> data(length_padded, width_padded);
    isce3::core::EArray2D<double> mask(length_padded, width_padded);

    // mask is one everywhere
    mask = 1.0;

    // create data. Note paddeed boundary is zero
    for (int line = pad_rows / 2; line < pad_rows / 2 + length; ++line) {
        for (int col = pad_cols / 2; col < pad_cols / 2 + width; ++col) {
            data(line, col) = create_data(line, col);
        }
    }

    // create the kernels
    isce3::core::EArray2D<double> kernelColumns(1, kernel_width);
    kernelColumns = 1.0 / 3.0;
    isce3::core::EArray2D<double> kernelRows(kernel_length, 1);
    kernelRows = 1.0 / 3.0;

    // two 1D Convolution in time domain and simultaneously decimate the output
    isce3::signal::convolve2D(filtered_data, data, mask, kernelColumns,
                              kernelRows);

    // convolution with box car kernel + decimation is equivalent to
    // multi-looking
    isce3::core::EArray2D<double> data_looked =
            isce3::signal::multilookAveraged(data.block(1, 1, length, width),
                                             kernel_length, kernel_width);

    // max error
    double max_err = 0.0;

    isce3::core::EArray2D<double> diff = data_looked - filtered_data;
    for (int line = 0; line < length_decimated; ++line) {
        for (int col = 0; col < width_decimated; ++col) {

            if (std::abs(diff(line, col)) > max_err)
                max_err = std::abs(diff(line, col));
        }
    }

    ASSERT_LT(max_err, 1.0e-12);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
