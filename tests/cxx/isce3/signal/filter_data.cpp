#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include <isce3/io/Raster.h>
#include <isce3/signal/filter2D.h>
#include <isce3/signal/filterKernel.h>

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

double product(std::valarray<double> x, std::valarray<double> y)
{
    x *= y;
    return x.sum();
}

std::complex<double> product_cpx(std::valarray<std::complex<double>> x,
                                 std::valarray<double> y)
{

    std::complex<double> sum = std::complex<double>(0.0, 0.0);
    for (size_t ii = 0; ii < x.size(); ++ii) {
        sum += x[ii] * y[ii];
    }
    return sum;
}

void write2raster(std::valarray<double> data, std::valarray<double> mask,
                  std::valarray<bool> indices, int width, int length)
{

    std::valarray<double> data_not_padded(length * width);
    std::valarray<double> mask_not_padded(length * width);

    data_not_padded = data[indices];
    mask_not_padded = mask[indices];

    isce3::io::Raster dataRaster("input_data_real", width, length, 1,
                                 GDT_Float64, "ENVI");
    isce3::io::Raster maskRaster("input_mask_real", width, length, 1,
                                 GDT_Float64, "ENVI");

    dataRaster.setBlock(data_not_padded, 0, 0, width, length);
    maskRaster.setBlock(mask_not_padded, 0, 0, width, length);
}

void write2raster_cpx(std::valarray<std::complex<double>> data,
                      std::valarray<double> mask, std::valarray<bool> indices,
                      int width, int length)
{

    std::valarray<std::complex<double>> data_not_padded(length * width);
    data_not_padded = data[indices];
    isce3::io::Raster dataRaster("input_data_cpx", width, length, 1,
                                 GDT_CFloat64, "ENVI");
    isce3::io::Raster maskRaster("input_mask_cpx", width, length, 1,
                                 GDT_Float64, "ENVI");

    dataRaster.setBlock(data_not_padded, 0, 0, width, length);
    maskRaster.setBlock(mask, 0, 0, width, length);
}

TEST(FilterData, FilterRealData)
{

    int length = 20;
    int width = 31;

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
    std::valarray<double> data(length_padded * width_padded);
    std::valarray<double> mask(length * width);
    std::valarray<bool> indices((length_padded) * (width_padded));

    // mask is one everywhere
    mask = 1.0;
    indices = false;

    // create data. Note paddeed boundary is zero
    for (int line = pad_rows / 2; line < pad_rows / 2 + length; ++line) {
        for (int col = pad_cols / 2; col < pad_cols / 2 + width; ++col) {

            data[line * width_padded + col] = create_data(line, col);
            indices[line * width_padded + col] = true;
        }
    }

    write2raster(data, mask, indices, width, length);

    isce3::io::Raster filtDataRaster("output.filtered_data", width, length, 1,
                                     GDT_Float64, "ENVI");

    // create the kernels
    // 1D kernel in columns
    std::valarray<double> kernelColumns =
            isce3::signal::boxcar1D(kernel_width); //, kernelColumns);

    // 1D kernel in rows direction
    std::valarray<double> kernelRows =
            isce3::signal::boxcar1D(kernel_length); // , kernelRows);

    // A 2D kernel
    std::valarray<double> kernel2D =
            isce3::signal::boxcar2D(kernel_width, kernel_length); //, kernel2D);

    isce3::io::Raster dataRaster("input_data_real");
    isce3::io::Raster maskRaster("input_mask_real");

    isce3::signal::filter2D<double>(filtDataRaster, dataRaster, kernelColumns,
                                    kernelRows);

    // error
    double max_err = 0.0;

    filtDataRaster.getBlock(filtered_data, 0, 0, width, length);
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
            double result = product(d, kernel2D);

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

TEST(FilterData, FilterComplexData)
{

    int length = 200;
    int width = 311;

    int kernel_width = 3;
    int kernel_length = 3;

    // number of pixels padded
    int pad_cols = kernel_width - 1;
    int pad_rows = kernel_length - 1;

    // width and length of the padded buffers
    int width_padded = width + pad_cols;
    int length_padded = length + pad_rows;

    // buffer for data after convolution
    std::valarray<std::complex<double>> filtered_data(length * width);

    // buffer for padded data and mask
    std::valarray<std::complex<double>> data(length_padded * width_padded);
    std::valarray<double> mask(length * width);
    std::valarray<bool> indices((length_padded) * (width_padded));

    // mask is one everywhere
    mask = 1.0;
    indices = false;

    // create data. Note paddeed boundary is zero
    for (int line = pad_rows / 2; line < pad_rows / 2 + length; ++line) {
        for (int col = pad_cols / 2; col < pad_cols / 2 + width; ++col) {

            data[line * width_padded + col] = std::complex<double>(
                    std::cos(line * col), std::sin(line * col));
            indices[line * width_padded + col] = true;
        }
    }

    write2raster_cpx(data, mask, indices, width, length);

    isce3::io::Raster filtDataRaster("output_cpx.filtered_data", width, length,
                                     1, GDT_CFloat64, "ENVI");
    // create the kernels
    // 1D kernel in columns
    std::valarray<double> kernelColumns = isce3::signal::boxcar1D(kernel_width);

    // 1D kernel in rows direction
    std::valarray<double> kernelRows = isce3::signal::boxcar1D(kernel_length);

    // A 2D kernel
    std::valarray<double> kernel2D =
            isce3::signal::boxcar2D(kernel_width, kernel_length);

    isce3::io::Raster dataRaster("input_data_cpx");
    isce3::io::Raster maskRaster("input_mask_cpx");

    int block_rows = 20;
    isce3::signal::filter2D<std::complex<double>>(
            filtDataRaster, dataRaster, kernelColumns, kernelRows, block_rows);

    // error
    double max_err = 0.0;
    filtDataRaster.getBlock(filtered_data, 0, 0, width, length);
    // buffer for data when kernel is centered on the pixel of interest
    std::valarray<std::complex<double>> d(kernel_width * kernel_length);
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
            std::complex<double> result = product_cpx(d, kernel2D);

            // compute the difference
            double diff = std::arg(filtered_data[(line - pad_rows / 2) * width +
                                                 col - pad_cols / 2] *
                                   std::conj(result));
            if (diff > max_err)
                max_err = diff;
        }
    }

    ASSERT_LT(max_err, 1.0e-12);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
