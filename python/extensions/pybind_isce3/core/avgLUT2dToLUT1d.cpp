#include "avgLUT2dToLUT1d.h"

#include <isce3/core/LUT1d.h>
#include <isce3/core/LUT2d.h>

namespace py = pybind11;

void addbinding_avgLUT2dToLUT1d(py::module & m)
{
    m.def("avg_lut2d_to_lut1d", &isce3::core::avgLUT2dToLUT1d<double>,
            py::arg("lut2d"),
            py::arg("axis")=0,
        R"(LUT1d made by averaging LUT2d along rows or columns.

        Parameters
        ----------
        lut2d : LUT2d
                LUT2d to be converted to LUT1d
        axis : int
                Axis to average along. 0 for rows and 1 for columns.)");
}
