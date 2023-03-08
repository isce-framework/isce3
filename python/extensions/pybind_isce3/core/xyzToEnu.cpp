#include "avgLUT2dToLUT1d.h"

#include <pybind11/eigen.h>

#include <isce3/core/DenseMatrix.h>

namespace py = pybind11;

void addbinding_xyzToEnu(py::module & m)
{
    m.def("xyz_to_enu", &isce3::core::Mat3::xyzToEnu,
            py::arg("lat"),
            py::arg("lon"),
        R"(Compute ENU basis transformation matrix
     *  @param[in] lat
     *  @param[in] lon Longitude in radians
     *  @param[out] enumat Matrix with rotation matrix */

        Parameters
        ----------
        lat: float
            Latitude in radians
        lon: float
            Longitude in radians

        Returns
        -------
        _: np.array
            Matrix with rotation matrix)");
}
