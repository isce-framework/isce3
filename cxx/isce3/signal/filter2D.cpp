#include "filter2D.h"

#include <complex>
#include <iostream>

#include <isce3/core/TypeTraits.h>
#include <isce3/core/Utilities.h>
#include <isce3/except/Error.h>
#include <isce3/io/Raster.h>
#include <isce3/signal/convolve.h>
#include <isce3/signal/decimate.h>

void check_kernels(const std::valarray<double>& kernel_columns,
                   const std::valarray<double>& kernel_rows)
{

    size_t ncols_kernel = kernel_columns.size();
    if (ncols_kernel % 2 == 0) {
        std::cout << "Warning: the size of the kernel in columns (X) direction "
                     "is even"
                  << std::endl;
        std::cout << "Warning: an even kernel shifts the grid by half a pixel."
                  << std::endl;
    }

    size_t nrows_kernel = kernel_rows.size();
    if (nrows_kernel % 2 == 0) {
        std::cout << "Warning: the size of the kernel in rows (Y) direction is "
                     "even."
                  << std::endl;
        std::cout << "Warning: an even kernel shifts the grid by half a pixel."
                  << std::endl;
    }
}

void check_rasters(isce3::io::Raster& input_raster,
                   isce3::io::Raster& output_raster,
                   isce3::io::Raster& mask_raster, const bool do_decimate,
                   const int ncols_kernel, const int nrows_kernel,
                   const bool mask_data)
{

    // sanity checks
    if (!do_decimate && input_raster.width() != output_raster.width()) {
        std::string errmsg = "input and output rasters must have same width";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    if (!do_decimate && input_raster.length() != output_raster.length()) {
        std::string errmsg = "input and output rasters must have same length";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    if (do_decimate) {
        size_t out_width = input_raster.width() / ncols_kernel;
        size_t out_length = input_raster.length() / nrows_kernel;

        if (output_raster.width() != out_width) {
            std::string errmsg = "output raster's width does not match the "
                                 "input raster's width divided by kernel width";
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
        }

        if (output_raster.length() != out_length) {
            std::string errmsg =
                    "output raster's length does not match the input raster's "
                    "length divided by kernel length ";
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
        }
    }

    if (input_raster.width() != mask_raster.width() && mask_data) {
        std::string errmsg = "input and mask rasters must have the same width";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    if (input_raster.length() != mask_raster.length() && mask_data) {
        std::string errmsg = "input and mask rasters must have the same length";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }
}

void setup_block_parameters(const int nrows, const int blockRows,
                            const int row_start, const int nblocks,
                            const int block, const int pad_rows,
                            int& block_line_start, int& line_start_read,
                            int& line_stop_read, int& block_rows_data,
                            int& block_rows_data_padded)
{

    if ((row_start + blockRows) > nrows) {
        block_rows_data = nrows - row_start;
    } else {
        block_rows_data = blockRows;
    }

    // Due to padding required for convolution, we need to know
    // where reading the block of data starts/stops and where the
    // block of data within the padded block is located
    if (block == 0 && nblocks == 1) {
        // when there is only one block
        line_start_read = row_start;
        line_stop_read = line_start_read + block_rows_data;
        block_line_start = pad_rows;

    } else if (block == 0) {
        // at first block and there are more than one blocks to process
        line_start_read = row_start;
        line_stop_read = line_start_read + block_rows_data + pad_rows;
        block_line_start = pad_rows;

    } else if (block == nblocks - 1) {
        // at last block and there more than one block
        line_start_read = row_start - pad_rows;
        line_stop_read = line_start_read + block_rows_data + pad_rows;
        block_line_start = 0;

    } else {
        // any block in the middle
        line_start_read = row_start - pad_rows;
        line_stop_read = line_start_read + block_rows_data + 2 * pad_rows;
        block_line_start = 0;
    }

    // number of lines in the block
    block_rows_data_padded = line_stop_read - line_start_read;
}

template<typename T>
void isce3::signal::filter2D(isce3::io::Raster& output_raster,
                             isce3::io::Raster& input_raster,
                             const std::valarray<double>& kernel_columns,
                             const std::valarray<double>& kernel_rows, int block_rows)
{

    bool do_decimate = false;
    if (input_raster.width() != output_raster.width()) {
        do_decimate = true;
    }
    if (input_raster.length() != output_raster.length()) {
        do_decimate = true;
    }

    // sanity checks
    if (kernel_columns.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "1D Kernel in columnss direction should have non-zero size");
    }
    if (kernel_rows.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "1D Kernel in rows direction should have non-zero size");
    }

    bool mask_data = false;
    std::string vsimem_ref = (
        "/vsimem/" + getTempString("filter_2d"));
    isce3::io::Raster mask_raster(vsimem_ref, 1, 1, 1, GDT_Float32,
                                  "ENVI");

    filter2D<T>(output_raster, input_raster, mask_raster, kernel_columns,
                kernel_rows, do_decimate, mask_data, block_rows);
}

template<typename T>
void isce3::signal::filter2D(isce3::io::Raster& output_raster,
                             isce3::io::Raster& input_raster,
                             isce3::io::Raster& mask_raster,
                             const std::valarray<double>& kernel_columns,
                             const std::valarray<double>& kernel_rows, int block_rows)
{

    std::cout << "A mask is provided. The input will be masked before filtering"
              << std::endl;
    bool mask_data = true;

    bool do_decimate = false;
    if (input_raster.width() != output_raster.width()) {
        do_decimate = true;
    }
    if (input_raster.length() != output_raster.length()) {
        do_decimate = true;
    }

    filter2D<T>(output_raster, input_raster, mask_raster, kernel_columns,
                kernel_rows, do_decimate, mask_data, block_rows);
}

template<typename T>
void isce3::signal::filter2D(isce3::io::Raster& output_raster,
                             isce3::io::Raster& input_raster,
                             isce3::io::Raster& mask_raster,
                             const std::valarray<double>& kernel_columns,
                             const std::valarray<double>& kernel_rows,
                             const bool do_decimate, const bool mask_data,
                             int block_rows)
{

    // sanity checks
    if (kernel_columns.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "1D Kernel in columnss direction should have non-zero size");
    }
    if (kernel_rows.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "1D Kernel in rows direction should have non-zero size");
    }

    int nrows_kernel_input = kernel_rows.size();
    int ncols_kernel_input = kernel_columns.size();

    check_kernels(kernel_columns, kernel_rows);

    int ncols_kernel = kernel_columns.size();
    int nrows_kernel = kernel_rows.size();

    check_rasters(input_raster, output_raster, mask_raster, do_decimate,
                  ncols_kernel_input, nrows_kernel_input, mask_data);

    int nrows = input_raster.length();
    int ncols = input_raster.width();

    int pad_rows = nrows_kernel / 2;
    int pad_cols = ncols_kernel / 2;

    int ncols_padded = ncols + 2 * pad_cols;

    block_rows = (block_rows / nrows_kernel) * nrows_kernel;

    // number of blocks to process
    int nblocks = nrows / block_rows;
    if (nblocks == 0) {
        nblocks = 1;
    } else if (nrows % (nblocks * block_rows) != 0) {
        nblocks += 1;
    }

    std::cout << "number of blocks: " << nblocks << std::endl;

    // buffer for a block of data
    // the buffer is padded for the input data and mask
    int block_rows_padded = block_rows + 2 * pad_rows;
    std::valarray<T> input(block_rows_padded * ncols_padded);
    std::valarray<T> output(block_rows * ncols);

    std::valarray<bool> mask(0);
    if (mask_data)
        mask.resize((block_rows_padded) *ncols_padded);

    std::valarray<T> output_decimated;
    size_t block_rows_decimated = block_rows / nrows_kernel_input;
    size_t ncols_decimated = ncols / ncols_kernel_input;

    if (do_decimate) {
        output_decimated.resize(block_rows_decimated * ncols_decimated);
    }

    for (int block = 0; block < nblocks; ++block) {
        std::cout << "working on block: " << block + 1 << std::endl;
        int row_start = block * block_rows;
        std::cout << "row_start: " << row_start << std::endl;
        // number of lines of data in this block. block_rows_data<= blockRows
        // Note that blockRows is fixed number of lines
        // block_rows_data might be less than or equal to blockRows.
        // e.g. if nrows = 512, and blockRows = 100, then
        // block_rows_data for last block will be 12
        int block_rows_data, block_rows_data_padded;
        int line_start_read, line_stop_read, block_line_start;
        setup_block_parameters(nrows, block_rows, row_start, nblocks, block,
                               pad_rows, block_line_start, line_start_read,
                               line_stop_read, block_rows_data,
                               block_rows_data_padded);

        // containers for one line of data
        std::valarray<T> data_line(ncols);

        input = 0.0;
        output = 0.0;

        // read the block of data
        for (size_t line = 0; line < block_rows_data_padded; ++line) {

            input_raster.getLine(data_line, line_start_read + line);
            input[std::slice((line + block_line_start) * ncols_padded +
                                     pad_cols,
                             ncols, 1)] = data_line;
        }

        if (mask_data) {
            mask = false;
            std::valarray<bool> mask_line(ncols);
            // read the block of data
            for (size_t line = 0; line < block_rows_data_padded; ++line) {

                mask_raster.getLine(mask_line, line_start_read + line);
                mask[std::slice((line + block_line_start) * ncols_padded +
                                        pad_cols,
                                ncols, 1)] = mask_line;
            }

            // Convolution in time domain
            isce3::signal::convolve2D(output, input, mask, kernel_columns,
                                      kernel_rows, ncols, ncols_padded);

        } else {
            // Convolution in time domain
            isce3::signal::convolve2D(output, input, kernel_columns,
                                      kernel_rows, ncols, ncols_padded);
        }

        // write the output block of filtered data to the raster
        if (do_decimate) {
            size_t rows_offset = nrows_kernel / 2;
            size_t cols_offset = ncols_kernel / 2;

            isce3::signal::decimate(output_decimated, output, block_rows, ncols,
                                    block_rows_decimated, ncols_decimated,
                                    nrows_kernel_input, ncols_kernel_input,
                                    rows_offset, cols_offset);

            output_raster.setBlock(output_decimated, 0,
                                   row_start / nrows_kernel_input,
                                   ncols / ncols_kernel_input,
                                   block_rows_data / nrows_kernel_input);

        } else {
            output_raster.setBlock(output, 0, row_start, ncols,
                                   block_rows_data);
        }
    }
}

#define SPECIALIZE_FILTER(T)                                                   \
    template void isce3::signal::filter2D<T>(                                  \
            isce3::io::Raster & output_raster,                                 \
            isce3::io::Raster & input_raster,                                  \
            const std::valarray<double> & kernel_columns,                      \
            const std::valarray<double> & kernel_rows, int block_rows);        \
    template void isce3::signal::filter2D<T>(                                  \
            isce3::io::Raster & output_raster,                                 \
            isce3::io::Raster & input_raster, isce3::io::Raster & mask_raster, \
            const std::valarray<double> & kernel_columns,                      \
            const std::valarray<double> & kernel_rows, int block_rows);        \
    template void isce3::signal::filter2D<T>(                                  \
            isce3::io::Raster & output_raster,                                 \
            isce3::io::Raster & input_raster, isce3::io::Raster & mask_raster, \
            const std::valarray<double> & kernel_columns,                      \
            const std::valarray<double> & kernel_rows, const bool do_decimate, \
            const bool mask, int block_rows)

SPECIALIZE_FILTER(float);
SPECIALIZE_FILTER(std::complex<float>);
SPECIALIZE_FILTER(double);
SPECIALIZE_FILTER(std::complex<double>);
