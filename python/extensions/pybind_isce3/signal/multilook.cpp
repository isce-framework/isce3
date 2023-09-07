#include "multilook.h"

#include <isce3/core/EMatrix.h>
#include <isce3/signal/multilook.h>

#include <complex>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using isce3::core::EArray2D;

template<typename EigenInputType>
void addbinding_multilook(py::module& m)
{
    // Expose multilookSummed function
    m.def("multilook_summed", [](
        const EigenInputType& input,
        int row_looks,
        int col_looks) {
            return isce3::signal::multilookSummed(input, row_looks, col_looks);
        },
        py::arg("input"),
        py::arg("row_looks"),
        py::arg("col_looks"),
        R"(
        Multilooks an input 2D array by summing contributions to each pixel.

        When the length (width) of the input array is not an integer
        multiple of the number of row_looks (col_looks), the length (width)
        of the output array gets truncated i.e., the fractional part of the
        division gets discarded.

        Parameters
        ----------
        input: numpy.ndarray
            The input array to multilook
        row_looks: int
            The number of looks in the vertical direction
        col_looks: int
            The number of looks in the horizontal direction

        Returns
        -------
        numpy.ndarray
            The multilooked output
        )");

    // Expose the multilookAveraged function
    m.def("multilook_averaged", [](
        const EigenInputType& input,
        int row_looks,
        int col_looks)
        {
            return isce3::signal::multilookAveraged(input, row_looks,
                    col_looks);
        },
        py::arg("input"),
        py::arg("row_looks"),
        py::arg("col_looks"),
        R"(
        Multilooks an input 2D array by averaging contributions to each pixel.

        When the length (width) of the input array is not an integer
        multiple of the number of row_looks (col_looks), the length (width)
        of the output array gets truncated i.e., the fractional part of the
        division gets discarded.

        Parameters
        ----------
        input: numpy.ndarray
            The input array to multilook
        row_looks: int
            The number of looks in the vertical direction
        col_looks: int
            The number of looks in the horizontal direction

        Returns
        -------
        numpy.ndarray
            The multilooked output
        )");

    // Expose the multilookNoData function
    m.def("multilook_nodata", [](
        const EigenInputType& input,
        int row_looks,
        int col_looks,
        const typename EigenInputType::value_type nodata)
        {
            return isce3::signal::multilookNoData(input, row_looks, col_looks,
                    nodata);
        },
        py::arg("input"),
        py::arg("row_looks"),
        py::arg("col_looks"),
        py::arg("nodata"),
        R"(
        Multilooks an input 2D array by taking the average of contributions to
        each pixel, while masking out a provided constant no-data value.

        When the length (width) of the input array is not an integer
        multiple of the number of row_looks (col_looks), the length (width)
        of the output array gets truncated i.e., the fractional part of the
        division gets discarded.

        Parameters
        ----------
        input: numpy.ndarray
            The input array to multilook
        row_looks: int
            The number of looks in the vertical direction
        col_looks: int
            The number of looks in the horizontal direction
        nodata: input.dtype
            The value to be masked

        Returns
        -------
        numpy.ndarray
            The multilooked output
        )");
}

template void addbinding_multilook<EArray2D<float>>(
    py::module& m);
template void addbinding_multilook<EArray2D<std::complex<float>>>(
    py::module& m);
template void addbinding_multilook<EArray2D<double>>(
    py::module& m);
template void addbinding_multilook<EArray2D<std::complex<double>>>(
    py::module& m);
