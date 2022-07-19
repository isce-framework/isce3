#include "symmetrize.h"

#include <isce3/io/Raster.h>

namespace py = pybind11;

void addbinding_symmetrize(pybind11::module& m)
{
    m.def("symmetrize_cross_pol_channels",
            &isce3::polsar::symmetrizeCrossPolChannels, py::arg("hv_raster"),
            py::arg("vh_raster"), py::arg("output_raster"),
            py::arg("memory_mode") = isce3::core::MemoryModeBlocksY::AutoBlocksY,
            py::arg("hv_raster_band") = 1, py::arg("vh_raster_band") = 1,
            py::arg("output_raster_band") = 1,
            R"(Symmetrize cross-polarimetric channels.

           The current implementation considers that the cross-polarimetric 
           channels (HV and VH) are already calibrated so the polarimetric
           symmetrization is represented by the simple arithmetic average.

          Parameters
          ---------
          hv_raster : isce3.io.Raster
              Raster containing the HV polarization channel
          vh_raster : isce3.io.Raster
              Raster containing the VH polarization channel
          output_raster : isce3.io.Raster
              Output symmetrized raster
          memory_mode : isce3.core.MemoryModeBlocksY, optional
              Select memory mode
          hv_raster_band : int
              Band (starting from 1) containing the HV polarization channel
              within `hv_raster`
          vh_raster_band : int
              Band (starting from 1) containing the VH polarization channel
              within `hv_raster`
          output_raster_band : int
              Band (starting from 1) that will contain the symmetrized 
              cross-polarimetric channel
          )");
}
