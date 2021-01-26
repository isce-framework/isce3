#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include <isce3/signal/decimate.h>

double create_data(int line, int col) { return line + col; }

TEST(Decimate, Decimate)
{

    size_t width = 10;
    size_t length = 101;

    size_t decimation_cols = 4;
    size_t decimation_rows = 5;

    size_t rows_offset = 0;
    size_t cols_offset = 0;

    size_t width_decimated = (width - cols_offset - 1) / decimation_cols + 1;
    size_t length_decimated = (length - rows_offset - 1) / decimation_rows + 1;

    // reserve memory for a block of data with the size of nfft
    std::valarray<double> data(width * length);
    std::valarray<double> dataDecimated(width_decimated * length_decimated);
    std::valarray<double> expectedResults(width_decimated * length_decimated);

    // a simple band limited signal (a linear phase ramp)
    for (size_t line = 0; line < length; ++line) {
        for (size_t col = 0; col < width; ++col) {

            data[line * width + col] = create_data(line, col);
        }
    }

    // a simple band limited signal (a linear phase ramp)
    for (size_t line = 0; line < length_decimated; ++line) {
        for (size_t col = 0; col < width_decimated; ++col) {

            expectedResults[line * width_decimated + col] =
                    create_data(line * decimation_rows, col * decimation_cols);
        }
    }

    isce3::signal::decimate(dataDecimated, data, length, width,
                            length_decimated, width_decimated, decimation_rows,
                            decimation_cols, rows_offset, cols_offset);

    // max error tolerance
    double max_err = 0.0;
    for (size_t i = 0; i < width_decimated * length_decimated; ++i) {

        double diff = dataDecimated[i] - expectedResults[i];

        // compare the phase of the difference with the max_error
        if (std::abs(diff) > max_err)
            max_err = std::abs(diff);
    }

    ASSERT_LT(max_err, 1.0e-14);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
