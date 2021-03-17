#include "convolve2D.h"

#include <isce3/except/Error.h>
#include <isce3/signal/convolve.h>

#include <complex>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using isce3::signal::convolve2D;
using isce3::core::EArray2D;

template<typename T>
void addbinding_convolve2D(py::module& m)
{
    m
    .def("convolve2D", [](
        const isce3::core::EArray2D<T>& input,
        const isce3::core::EArray2D<double>& kernel_columns,
        const isce3::core::EArray2D<double>& kernel_rows,
        const bool decimate)
        {
            // get input dimensions
            auto input_length = input.rows();
            auto input_width = input.cols();

            // check kernel_columns dimensions
            auto kernel_cols_length = kernel_columns.rows();
            auto kernel_cols_width = kernel_columns.cols();
            if (kernel_cols_length != 1)
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                                  "kernel column not 1 x N");
            if (kernel_cols_width > input_width)
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                                  "kernel column width > input width");

            // check kernel_rows dimensions
            auto kernel_rows_width = kernel_rows.cols();
            if (kernel_rows_width != 1)
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                                  "kernel column not N x 1");
            if (kernel_rows_width > input_width)
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                                  "kernel rows width > input length");

            // determine pad size
            // divide 2 in () to ensure integer division done 1st
            auto length_pad = 2 * (kernel_rows.size() / 2);
            auto width_pad = 2 * (kernel_columns.size() / 2);

            // get output dimensions
            auto output_length = input_length - length_pad;
            auto output_width = input_width - width_pad;
            if (decimate) {
                output_length /= kernel_rows.size();
                output_width /= kernel_columns.size();
            }

            // init weight of ones
            isce3::core::EArray2D<double> weights;
            weights.setOnes(input_length, input_width);

            // init output
            isce3::core::EArray2D<T> output(output_length, output_width);

            // convolve/decimate
            convolve2D<T>(output, input, weights, kernel_columns, kernel_rows);

            return output;
        },
        py::arg("input"),
        py::arg("kernel_columns"),
        py::arg("kernel_rows"),
        py::arg("decimate"),
        "2D convolution in time domain with separable kernels")
    .def("convolve2D", [](
        const isce3::core::EArray2D<T>& input,
        const isce3::core::EArray2D<double>& weights,
        const isce3::core::EArray2D<double>& kernel_columns,
        const isce3::core::EArray2D<double>& kernel_rows,
        const bool decimate)
        {
            // get input dimensions
            auto input_length = input.rows();
            auto input_width = input.cols();

            // get weight dimensions
            auto weight_length = weights.rows();
            auto weight_width = weights.cols();

            // ensure input and weight dimensions match
            if (input_length != weight_length || input_width != weight_width)
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                                  "input and weight dimensions do not match");

            // check kernel_columns dimensions
            auto kernel_cols_length = kernel_columns.rows();
            auto kernel_cols_width = kernel_columns.cols();
            if (kernel_cols_length != 1)
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                                  "kernel column not 1 x N");
            if (kernel_cols_width > input_width)
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                                  "kernel column width > input width");

            // check kernel_rows dimensions
            auto kernel_rows_width = kernel_rows.cols();
            if (kernel_rows_width != 1)
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                                  "kernel column not N x 1");
            if (kernel_rows_width > input_width)
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                                  "kernel rows width > input length");

            // determine pad size
            auto length_pad = 2 * (kernel_rows.size() / 2);
            auto width_pad = 2 * (kernel_columns.size() / 2);

            // determine output dimensions
            auto output_length = input_length - length_pad;
            auto output_width = input_width - width_pad;
            if (decimate) {
                output_length /= kernel_rows.size();
                output_width /= kernel_columns.size();
            }
          
            // init output
            isce3::core::EArray2D<T> output(output_length, output_width);
          
            // convolve/decimate
            convolve2D<T>(output, input, weights, kernel_columns, kernel_rows);
          
            return output;
        },
        py::arg("input"),
        py::arg("weights"),
        py::arg("kernel_columns"),
        py::arg("kernel_rows"),
        py::arg("decimate"),
        "2D convolution in time domain with separable kernels and mask/weights")
;
}

template void addbinding_convolve2D<float>(py::module& m);
template void addbinding_convolve2D<double>(py::module& m);
template void addbinding_convolve2D<std::complex<float>>(py::module& m);
template void addbinding_convolve2D<std::complex<double>>(py::module& m);
