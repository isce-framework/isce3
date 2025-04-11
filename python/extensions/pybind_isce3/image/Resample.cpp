#include "Resample.h"

#include <isce3/core/EMatrix.h>
#include <isce3/core/LUT2d.h>
#include <isce3/image/Resample.h>
#include <isce3/product/RadarGridParameters.h>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void addbindings_resamp(py::module & m)
{
    // Write _resample_to_coords as a private function. This will be used by a wrapper
    // called resample_to_coords that is written in Python.
    m.def("_resample_to_coords",
        &isce3::image::v2::resampleToCoords,
        py::arg("output_data_block"),
        py::arg("input_data_block"),
        py::arg("range_input_indices"),
        py::arg("azimuth_input_indices"),
        py::arg("in_radar_grid"),
        py::arg("native_doppler"),
        py::arg("fill_value") =
            std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()),
        R"(
        Interpolate input SLC block into the index values of the output block.

        Parameters
        ----------
        out_data_block: numpy.ndarray (complex64)
            The output SLC array to modify.
        in_data_block: numpy.ndarray (complex64)
            Input SLC array in secondary coordinates.
        range_input_indices: numpy.ndarray (float64)
            The range (radar-coordinates x) index of the output pixels in the input
            grid.
        azimuth_input_indices: numpy.ndarray (float64)
            The azimuth (radar-coordinates y) index of the output pixels in the input
            grid.
        in_radar_grid: isce3.product.RadarGridParameters
            Radar grid parameters of the input SLC data.
        native_doppler: isce3.core.LUT2d
            2D LUT describing the native doppler of the input SLC image, in Hz.
        fill_value: complex
            The value to fill out-of-bounds pixels with. Defaults to NaN + j*NaN.
        )"
    );
}
