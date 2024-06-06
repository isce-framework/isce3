#include "Flatten.h"

#include <isce3/image/Flatten.h>
#include <isce3/product/RadarGridParameters.h>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


void addbindings_modulate(py::module & m)
{
    m.def(
        "flatten_at_coords",
        &isce3::image::flatten::flattenAtCoords,
        py::arg("data_block"),
        py::arg("range_indices"),
        py::arg("radar_grid_out"),
        py::arg("radar_grid_in"),
        py::arg("in_rg_first_pixel"),
        py::arg("out_rg_first_pixel"),
        R"(
        Re-flatten a grid of SLC data from its' original grid parameters into a new set
        of grid parameters.

        Parameters
        ----------
        data_block : np.ndarray (complex64)
            The SLC data to flatten
        range_indices : np.ndarray (float64)
            range index of each coordinate pixel in the data block in the coordinate
            system of the alternate radar grid
        radar_grid_out : isce3.product.RadarGridParameters
            radar grid parameters of the alternate grid
        radar_grid_in : isce3.product.RadarGridParameters
            radar grid parameters of the original grid
        in_rg_first_pixel : int
            range index of the first sample of the original grid
        out_rg_first_pixel : int
            range index of the first sample of the alternate grid
        )"
    );
}
