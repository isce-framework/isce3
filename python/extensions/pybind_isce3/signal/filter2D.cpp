#include "filter2D.h"

#include <gdal_priv.h>
#include <valarray>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <isce3/signal/filter2D.h>
#include <isce3/except/Error.h>
#include <isce3/io/Raster.h>

namespace py = pybind11;

using isce3::signal::filter2D;
using isce3::io::Raster;

void addbinding_filter2D(py::module& m)
{
    m
    .def("filter2D", [](
        Raster& output,
        Raster& input,
        const std::valarray<double> & kernel_columns,
        const std::valarray<double> & kernel_rows,
        int block_rows) {
            int band=1;
            auto in_type = input.dtype(band);
            auto out_type = output.dtype(band);
            // ensure input and output data type match
            if (in_type != out_type)
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                      "Input and output data type do not match");
            // check kernel_columns dimensions
            if (kernel_columns.size() > input.width())
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                       "kernel column width > input width");
            // check kernel_rows dimensions
            if (kernel_rows.size() > input.length())
               throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                       "kernel rows width > input length");
            if (output.length() != input.length())
                if (output.length() != input.length() / kernel_rows.size())
                    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                          "output length not equal to input length or not"
                          "equal to input length divided by kernel row size");
            if (output.width() != input.width())
                if (output.width() != output.width() / kernel_columns.size())
                    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                          "output width not equal to input width or not"
                          "equal to input width divided by kernel columns size");
            // ensure types match
            switch (in_type) {
            case GDT_Float32: filter2D<float>(output, input, kernel_columns,
                                      kernel_rows, block_rows);
                                return;
            case GDT_Float64: filter2D<double>(output, input, kernel_columns,
                                      kernel_rows, block_rows);
                                return;
            case GDT_CFloat32: filter2D<std::complex<float>>(output, input,
                                      kernel_columns, kernel_rows, block_rows);
                                return;
            case GDT_CFloat64: filter2D<std::complex<double>>(output, input,
                                      kernel_columns, kernel_rows, block_rows);
                                return;
            default: throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                    "unsupported GDAL datatype");
            }

        },
        py::arg("output"),
        py::arg("input"),
        py::arg("kernel_columns"),
        py::arg("kernel_rows"),
        py::arg("block_rows"),
        "Filter real or complex data by convolving two separable 1D kernels")
;
}
