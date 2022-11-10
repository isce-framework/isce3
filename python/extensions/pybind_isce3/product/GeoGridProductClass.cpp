#include "GeoGridProductClass.h"

#include <string>
#include <valarray>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <isce3/io/IH5.h>
#include <isce3/product/GeoGridProduct.h>

namespace py = pybind11;

using isce3::product::GeoGridProduct;

void addbinding(pybind11::class_<GeoGridProduct> & pyGeoGridProduct)
{
    pyGeoGridProduct.def(py::init([](const std::string &h5file)
                {
                    // open file
                    isce3::io::IH5File file(h5file);

                    // instantiate and load a product
                    isce3::product::GeoGridProduct product(file);

                    // return product object
                    return product;
                }),
                py::arg("h5file"))
        .def_property("lookside",
                py::overload_cast<>(&GeoGridProduct::lookSide, py::const_),
                py::overload_cast<isce3::core::LookSide>(&GeoGridProduct::lookSide))
    ;
}
