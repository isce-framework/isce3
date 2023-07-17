#include "multilook.h"

#include <isce3/core/EMatrix.h>
#include <isce3/signal/multilook.h>

#include <complex>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using isce3::core::EArray2D;

template<typename EigenInputType, typename EigenWeightType>
void addbinding_multilook(py::module& m)
{
    // Expose multilookSummed function
    m.def("multilook_summed",
        &isce3::signal::multilookSummed<EigenInputType>,
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
    m.def("multilook_averaged",
        &isce3::signal::multilookAveraged<EigenInputType>,
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
    m.def("multilook_nodata",
        &isce3::signal::multilookNoData<EigenInputType>,
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

template void addbinding_multilook<EArray2D<float>, EArray2D<float>>(
        py::module& m);
template void addbinding_multilook<EArray2D<std::complex<float>>,
         EArray2D<float>>(py::module& m);
template void addbinding_multilook<EArray2D<double>, EArray2D<double>>(
        py::module& m);
template void addbinding_multilook<EArray2D<std::complex<double>>,
         EArray2D<double>>(py::module& m);
