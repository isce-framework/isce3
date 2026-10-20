#include "geogrid_ecef_coords.h"

#include <isce3/core/Projections.h>
#include <isce3/except/Error.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/product/GeoGridParameters.h>

#include <pybind11/numpy.h>

namespace py = pybind11;
using isce3::geometry::DEMInterpolator;
using isce3::product::GeoGridParameters;
using isce3::core::makeProjection;
using isce3::except::InvalidArgument;

void addbinding_get_geogrid_ecef_coords(pybind11::module& m)
{
    m.def("get_geogrid_ecef_coords",
        [](const GeoGridParameters& grid, const DEMInterpolator& dem)
        {
            const long m = grid.length();
            const long n = grid.width();

            auto proj_in = makeProjection(grid.epsg());
            auto proj_out = makeProjection(4978);

            if (not dem.haveStats()) {
                throw InvalidArgument(ISCE_SRCINFO(),
                    "Input DEM does not have stats.");
            }

            const double h = dem.meanHeight();

            auto out = py::array_t<double>({m, n, 3L});
            auto r = out.mutable_unchecked<3>();

            #pragma omp parallel for collapse(2)
            for (long i = 0; i < m; ++i) {
                for (long j = 0; j < n; ++j) {
                    isce3::core::Vec3 pos, xyz;
                    pos[0] = grid.startX() + grid.spacingX() * j;
                    pos[1] = grid.startY() + grid.spacingY() * i;
                    pos[2] = h;
                    isce3::core::projTransform(proj_in.get(), proj_out.get(),
                        pos, xyz);
                    #pragma unroll
                    for (int k = 0; k < 3; ++k) {
                        r(i, j, k) = xyz[k];
                    }
                }
            }
            return out;
	    },
        py::arg("grid"),
        py::arg("dem")
    );
}
