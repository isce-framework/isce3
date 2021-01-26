#include "decimate.h"

#include <complex>

#include <isce3/except/Error.h>

template<typename T>
void isce3::signal::decimate(std::valarray<T>& out, const std::valarray<T>& in,
                             size_t nrows, size_t ncols, size_t nrows_decimated,
                             size_t ncols_decimated, size_t rows_decimation,
                             size_t cols_decimation, size_t rows_offset,
                             size_t cols_offset)
{

    // sanity checks
    if (nrows <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Number of rows of the input data should be > 0");
    }
    if (ncols <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Number of columns of the input should be > 0");
    }
    if (nrows_decimated <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Number of rows of the output should be > 0");
    }
    if (ncols_decimated <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Number of columns of the output should be > 0");
    }
    if (in.size() != nrows * ncols) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Input size does not match the input rows and columns");
    }
    if (out.size() != nrows_decimated * ncols_decimated) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Output size does not match the output rows and columns");
    }

    // check if the input nrows_decimated and ncols_decimated are correct
    size_t width_decimated = (ncols - cols_offset - 1) / cols_decimation + 1;
    size_t length_decimated = (nrows - rows_offset - 1) / rows_decimation + 1;

    if (ncols_decimated != width_decimated) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "The input number of columns in the decimated "
                                "data does not seem correct");
    }
    if (nrows_decimated != length_decimated) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "The input number of rows in the decimated "
                                "data does not seem correct");
    }

    _Pragma("omp parallel for") for (size_t kk = 0;
                                     kk < nrows_decimated * ncols_decimated;
                                     ++kk)
    {
        size_t line_out = kk / ncols_decimated;
        size_t col_out = kk % ncols_decimated;
        size_t line_in = line_out * rows_decimation + rows_offset;
        size_t col_in = col_out * cols_decimation + cols_offset;
        out[line_out * ncols_decimated + col_out] =
                in[line_in * ncols + col_in];
    }
}

#define SPECIALIZE_DECIMATE(T)                                                 \
    template void isce3::signal::decimate(                                     \
            std::valarray<T>& out, const std::valarray<T>& in, size_t nrows,   \
            size_t ncols, size_t nrows_decimated, size_t ncols_decimated,      \
            size_t rows_decimation, size_t cols_decimation,                    \
            size_t rows_offset, size_t cols_offset);

SPECIALIZE_DECIMATE(float)
SPECIALIZE_DECIMATE(double)
SPECIALIZE_DECIMATE(std::complex<float>)
SPECIALIZE_DECIMATE(std::complex<double>)
