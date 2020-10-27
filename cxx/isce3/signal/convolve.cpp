#include "convolve.h"

#include <isce3/core/Constants.h>
#include <isce3/core/EMatrix.h>
#include <isce3/core/TypeTraits.h>
#include <isce3/core/Utilities.h>
#include <isce3/except/Error.h>

template<typename T>
void isce3::signal::convolve2D(std::valarray<T>& output,
                               const std::valarray<T>& input,
                               const std::valarray<double>& kernelColumns,
                               const std::valarray<double>& kernelRows,
                               int ncols, int ncols_padded)
{

    // sanity checks
    if (ncols <= 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Number of columns should be > 0");
    }
    if (ncols_padded <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Number of columns for padded data should be > 0");
    }
    if (kernelColumns.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "1D Kernel in columns direction should have non-zero size");
    }
    if (kernelRows.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "1D Kernel in rows direction should have non-zero size");
    }
    if (output.size() == 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Output should have non-zero size");
    }
    if (input.size() == 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Input should have non-zero size");
    }

    // a buffer for weights and fill with 1.0
    std::valarray<double> weights(1.0, input.size());
    convolve2D(output, input, weights, kernelColumns, kernelRows, ncols,
               ncols_padded);
}

template<typename T>
void isce3::signal::convolve2D(std::valarray<T>& output,
                               const std::valarray<T>& input,
                               const T& noDataValue,
                               const std::valarray<double>& kernelColumns,
                               const std::valarray<double>& kernelRows,
                               int ncols, int ncols_padded)
{

    // sanity checks
    if (ncols <= 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Number of columns should be > 0");
    }
    if (ncols_padded <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Number of columns for padded data should be > 0");
    }
    if (kernelColumns.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "1D Kernel in columns direction should have non-zero size");
    }
    if (kernelRows.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "1D Kernel in rows direction should have non-zero size");
    }
    if (output.size() == 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Output should have non-zero size");
    }
    if (input.size() == 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Input should have non-zero size");
    }

    std::valarray<bool> mask = isce3::core::makeMask(input, noDataValue);

    convolve2D(output, input, mask, kernelColumns, kernelRows, ncols,
               ncols_padded);
}

template<typename T>
void isce3::signal::convolve2D(std::valarray<T>& output,
                               const std::valarray<T>& input,
                               const std::valarray<bool>& mask,
                               const std::valarray<double>& kernelColumns,
                               const std::valarray<double>& kernelRows,
                               int ncols, int ncols_padded)
{

    // sanity checks
    if (ncols <= 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Number of columns should be > 0");
    }
    if (ncols_padded <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Number of columns for padded data should be > 0");
    }
    if (kernelColumns.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "1D Kernel in columns direction should have non-zero size");
    }
    if (kernelRows.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "1D Kernel in rows direction should have non-zero size");
    }
    if (output.size() == 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Output should have non-zero size");
    }
    if (input.size() == 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Input should have non-zero size");
    }
    if (mask.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Input mask should have non-zero size");
    }
    if (mask.size() != input.size()) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Input data and mask should have the same size");
    }

    // a buffer for weights
    std::valarray<double> weights(input.size());

    // fill the weights for pixels with valid mask with one
    weights[mask] = 1.0;

    convolve2D(output, input, weights, kernelColumns, kernelRows, ncols,
               ncols_padded);
}

template<typename T>
void isce3::signal::convolve2D(std::valarray<T>& output,
                               const std::valarray<T>& input,
                               const std::valarray<double>& weights,
                               const std::valarray<double>& kernelColumns,
                               const std::valarray<double>& kernelRows,
                               int ncols, int ncols_padded)
{

    int ncols_kernel = kernelColumns.size();
    int nrows_kernel = kernelRows.size();

    // sanity checks
    if (ncols <= 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Number of columns should be > 0");
    }
    if (ncols_padded <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Number of columns for padded data should be > 0");
    }
    if (ncols_kernel <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Kernel's number of columns should be > 0");
    }
    if (nrows_kernel <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Kernel's number of rows should be > 0");
    }
    if (output.size() == 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Output should have non-zero size");
    }
    if (input.size() == 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Input should have non-zero size");
    }
    if (weights.size() == 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Input mask should have non-zero size");
    }
    if (weights.size() != input.size()) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Input data and weights should have the same size");
    }

    // the start line of the block within the padded block
    int line_start = nrows_kernel / 2;

    // the start column of the block within the padded block
    int col_start = ncols_kernel / 2;

    // half of the kernel in columns
    int kernel_cols_half = ncols_kernel / 2;

    // half of the kernel in rows
    int kernel_rows_half = nrows_kernel / 2;

    // number of rows of the padded block
    int nrows_padded = input.size() / ncols_padded;

    // number of rows of the block without padding
    int nrows = output.size() / ncols;

    // container for the result of 1D convolution in columns direction
    std::valarray<T> tempOutput(nrows_padded * ncols_padded);

    std::valarray<double> tempWeights(nrows_padded * ncols_padded);
    // initialize the container
    tempOutput = 0;
    tempWeights = 0;

    using PromotedType = typename isce3::core::double_promote<T>::type;

    // convolve the 1D kernel in columns direction
    _Pragma("omp parallel for") for (size_t kk = 0; kk < nrows_padded * ncols;
                                     ++kk)
    {

        size_t line = kk / ncols;
        size_t col = kk % ncols + col_start;
        // multiply the kernel elements by the window of data centered on line,
        // col
        T sum = 0.0;
        double sum_kernel = 0.0;
        int center = line * ncols_padded + col;
        for (int j = 0; j < ncols_kernel; j++) {
            int window_col = col - kernel_cols_half + j;
            int window_element = line * ncols_padded + window_col;

            const auto element_value =
                    static_cast<PromotedType>(input[window_element]);
            sum += kernelColumns[j] * element_value * weights[window_element];

            sum_kernel += kernelColumns[j] * weights[window_element];
        }

        tempOutput[center] = sum;
        tempWeights[center] = sum_kernel;
    }

    // convolve the 1D kernel in rows direction
    _Pragma("omp parallel for") for (size_t kk = 0; kk < nrows * ncols; ++kk)
    {

        size_t line = kk / ncols + line_start;
        size_t col = kk % ncols + col_start;

        // multiply the kernel elements by the window of data centered on line,
        // col
        T s = 0.0;
        auto sum = static_cast<PromotedType>(s);
        double sum_kernel = 0.0;
        int center = line * ncols_padded + col;
        if (weights[center] != 0) {
            for (int j = 0; j < nrows_kernel; j++) {
                int window_line = line - kernel_rows_half + j;
                int window_element = window_line * ncols_padded + col;

                const auto element_value =
                        static_cast<PromotedType>(tempOutput[window_element]);
                sum += kernelRows[j] * element_value;

                sum_kernel += kernelRows[j] * tempWeights[window_element];
            }

            if (sum_kernel > 0.0) {
                output[(line - line_start) * ncols + (col - col_start)] =
                        sum / sum_kernel;
            } else {
                output[(line - line_start) * ncols + (col - col_start)] = 0.0;
            }
        }
    }
}

template<typename T>
void isce3::signal::convolve2D(
        isce3::core::EArray2D<T>& output, const isce3::core::EArray2D<T>& input,
        const isce3::core::EArray2D<double>& weights,
        const isce3::core::EArray2D<double>& kernelColumns,
        const isce3::core::EArray2D<double>& kernelRows)
{

    auto nrows_kernel = kernelRows.rows();
    auto ncols_kernel = kernelColumns.cols();

    // sanity checks
    if (kernelRows.cols() != 1) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "The 1D kernel in rows direction is "
                                         "expcted to have only one column");
    }
    if (kernelColumns.rows() != 1) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "The 1D kernel in columns direction "
                                         "is expcted to have only one row");
    }

    // the start line of the block within the padded block
    int line_start = nrows_kernel / 2;

    // the start column of the block within the padded block
    int col_start = ncols_kernel / 2;

    // half of the kernel in columns
    int kernel_cols_half = ncols_kernel / 2;

    // half of the kernel in rows
    int kernel_rows_half = nrows_kernel / 2;

    // number of rows of the padded block
    auto nrows_padded = input.rows();
    auto ncols_padded = input.cols();

    // number of rows of the block without padding
    int nrows = output.rows();
    int ncols = output.cols();

    // container for the result of 1D convolution in columns direction
    isce3::core::EArray2D<T> tempOutput(nrows_padded, ncols);

    isce3::core::EArray2D<double> tempWeights(nrows_padded, ncols);

    // initialize the container
    tempOutput = 0;
    tempWeights = 0;

    using PromotedType = typename isce3::core::double_promote<T>::type;

    size_t col_decimate = ncols_kernel;
    size_t cols_offset = kernel_cols_half;
    if (output.cols() == (ncols_padded - kernel_cols_half * 2)) {
        col_decimate = 1;
        cols_offset = 0;
    }

    size_t line_decimate = nrows_kernel;
    size_t rows_offset = kernel_rows_half;
    if (output.rows() == (nrows_padded - kernel_rows_half * 2)) {
        line_decimate = 1;
        rows_offset = 0;
    }

    // convolve the 1D kernel in columns direction
    _Pragma("omp parallel for") for (size_t kk = 0; kk < nrows_padded * ncols;
                                     ++kk)
    {

        size_t line_out = kk / ncols;
        size_t col_out = kk % ncols;
        size_t line_in = line_out;
        size_t col_in = col_out * col_decimate + col_start + cols_offset;

        // multiply the kernel elements by the window of data centered on line,
        // col
        T sum = 0.0;
        double sum_kernel = 0.0;
        for (int j = 0; j < ncols_kernel; j++) {
            int window_col = col_in - kernel_cols_half + j;

            const auto element_value =
                    static_cast<PromotedType>(input(line_in, window_col));

            sum += kernelColumns(0, j) * element_value *
                   weights(line_in, window_col);

            sum_kernel += kernelColumns(0, j) * weights(line_in, window_col);
        }

        tempOutput(line_out, col_out) = sum;
        tempWeights(line_out, col_out) = sum_kernel;
    }

    // convolve the 1D kernel in rows direction
    _Pragma("omp parallel for") for (size_t kk = 0; kk < nrows * ncols; ++kk)
    {

        size_t line_out = kk / ncols;
        size_t col_out = kk % ncols;
        size_t line_in = line_out * line_decimate + line_start + rows_offset;
        size_t col_in = col_out;

        // multiply the kernel elements by the window of data centered on line,
        // col
        T s = 0.0;
        auto sum = static_cast<PromotedType>(s);
        double sum_kernel = 0.0;
        if (weights(line_in, col_in) != 0) {
            for (int j = 0; j < nrows_kernel; j++) {
                int window_line = line_in - kernel_rows_half + j;

                const auto element_value = static_cast<PromotedType>(
                        tempOutput(window_line, col_in));
                sum += kernelRows(j, 0) * element_value;

                sum_kernel +=
                        kernelRows(j, 0) * tempWeights(window_line, col_in);
            }

            if (sum_kernel > 0.0) {
                output(line_out, col_out) = sum / sum_kernel;
            } else {
                output(line_out, col_out) = 0.0;
            }
        }
    }
}

#define SPECIALIZE_CONVOLVE(T)                                                 \
    template void isce3::signal::convolve2D(                                   \
            std::valarray<T>& output, const std::valarray<T>& input,           \
            const std::valarray<double>& weights,                              \
            const std::valarray<double>& kernelColumns,                        \
            const std::valarray<double>& kernelRows, int ncols,                \
            int ncols_padded);                                                 \
    template void isce3::signal::convolve2D(                                   \
            std::valarray<T>& output, const std::valarray<T>& input,           \
            const std::valarray<bool>& mask,                                   \
            const std::valarray<double>& kernelColumns,                        \
            const std::valarray<double>& kernelRows, int ncols,                \
            int ncols_padded);                                                 \
    template void isce3::signal::convolve2D(                                   \
            std::valarray<T>& output, const std::valarray<T>& input,           \
            const T& noData, const std::valarray<double>& kernelColumns,       \
            const std::valarray<double>& kernelRows, int ncols,                \
            int ncols_padded);                                                 \
    template void isce3::signal::convolve2D(                                   \
            std::valarray<T>& output, const std::valarray<T>& input,           \
            const std::valarray<double>& kernelColumns,                        \
            const std::valarray<double>& kernelRows, int ncols,                \
            int ncols_padded);                                                 \
    template void isce3::signal::convolve2D(                                   \
            isce3::core::EArray2D<T>& output,                                  \
            const isce3::core::EArray2D<T>& input,                             \
            const isce3::core::EArray2D<double>& weights,                      \
            const isce3::core::EArray2D<double>& kernelColumns,                \
            const isce3::core::EArray2D<double>& kernelRows);

SPECIALIZE_CONVOLVE(double)
SPECIALIZE_CONVOLVE(std::complex<double>)
SPECIALIZE_CONVOLVE(float)
SPECIALIZE_CONVOLVE(std::complex<float>)
