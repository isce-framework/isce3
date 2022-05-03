#include "relocateRaster.h"

#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>

namespace py = pybind11;

void addbinding_relocate_raster(pybind11::module& m)
{

    m.def("relocate_raster", &isce3::geogrid::relocateRaster,
          py::arg("input_raster"),
          py::arg("geogrid"), 
          py::arg("output_raster"),
          py::arg("interp_method") = isce3::core::BIQUINTIC_METHOD,
          R"(Relocate raster

             Relocate (reproject/resample) a raster over a given geogrid.
             The output raster is expected to have the same length & width
             as the specified geogrid, and the same number of bands as the
             input raster. Invalid pixels are filled with NaN values.

             Parameters
             ----------

             input_raster : isce3.io.Raster
                 Input raster
             geogrid : isce3.product.GeoGridParameters
                 Geogrid to be used as reference for output raster
             output_raster : isce3.io.Raster
                 Output raster
             interp_method :  isce3::core::dataInterpMethod
                 Interpolation method
)");

}
