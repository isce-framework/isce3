#include "LUT2d.h"

#include <isce3/core/Constants.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/Matrix.h>
#include <isce3/core/Serialization.h>
#include <isce3/except/Error.h>

#include <cstring>
#include <string>
#include <valarray>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using isce3::core::LUT2d;

static dataInterpMethod duck_method(py::object method)
{
    using isce3::core::dataInterpMethod;
    using isce3::core::parseDataInterpMethod;
    if (py::isinstance<py::str>(method)) {
        return parseDataInterpMethod(py::str(method));
    } else if (py::isinstance<dataInterpMethod>(method)) {
        return method.cast<dataInterpMethod>();
    } else {
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                                            "invalid type for interp method");
    }
}

template<typename T>
void addbinding(py::class_<LUT2d<T>> &pyLUT2d)
{

    pyLUT2d
        .def(py::init<>())
        .def(py::init([](double xstart, double ystart,
                    double dx, double dy,
                    py::array_t<T, py::array::c_style | py::array::forcecast> & py_data,
                    py::object method, bool b_error)
                {
                    // memcpy because ndarray not auto converted to eigen type for some reason
                    if (py_data.ndim() != 2) {
                        throw isce3::except::RuntimeError(ISCE_SRCINFO(), "buffer object must be 2-D");
                    }
                    isce3::core::Matrix<T> data(py_data.shape()[0], py_data.shape()[1]);
                    std::memcpy(data.data(), py_data.data(), py_data.nbytes());

                    // get interp method
                    auto interp_method = duck_method(method);

                    // return LUT2d object
                    return isce3::core::LUT2d<T>(xstart, ystart, dx, dy, data, interp_method, b_error);
                }),
                py::arg("xstart"),
                py::arg("ystart"),
                py::arg("dx"),
                py::arg("dy"),
                py::arg("data"),
                py::arg("method")="bilinear",
                py::arg("b_error")=true)
        .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> & py_xcoord,
                        py::array_t<double, py::array::c_style | py::array::forcecast> & py_ycoord,
                        py::array_t<T, py::array::c_style | py::array::forcecast> & py_data,
                        py::object method, bool b_error)
                {
                    // memcpy ndarrays because they're not auto converted to eigen type for some reason
                    std::valarray<double> xcoord(py_xcoord.size());
                    std::memcpy(&xcoord[0], py_xcoord.data(), py_xcoord.nbytes());

                    std::valarray<double> ycoord(py_ycoord.size());
                    std::memcpy(&ycoord[0], py_ycoord.data(), py_ycoord.nbytes());

                    if (py_data.ndim() != 2) {
                        throw isce3::except::RuntimeError(ISCE_SRCINFO(), "buffer object must be 2-D");
                    }
                    isce3::core::Matrix<T> data(py_data.shape()[0], py_data.shape()[1]);
                    std::memcpy(data.data(), py_data.data(), py_data.nbytes());

                    // get interp method
                    auto interp_method = duck_method(method);

                    // return LUT2d object
                    return isce3::core::LUT2d<T>(xcoord, ycoord, data, interp_method, b_error);
                }),
                py::arg("xcoord"),
                py::arg("ycoord"),
                py::arg("data"),
                py::arg("method")="bilinear",
                py::arg("b_error")=true)


        .def_static("load_from_h5", [](py::object h5py_group,
                                       const std::string& dataset_name) {

                auto id = h5py_group.attr("id").attr("id").cast<hid_t>();
                isce3::io::IGroup group(id);

                LUT2d<T> lut;
                isce3::core::loadCalGrid(group, dataset_name, lut);

                return lut;
            },
            "De-serialize LUT from h5py.Group object",
            py::arg("h5py_group"),
            py::arg("dataset_name"))

        .def("save_to_h5", [](const LUT2d<T>& self, py::object h5py_group,
                              const std::string& name,
                              const isce3::core::DateTime& epoch,
                              const std::string& units) {

                auto id = h5py_group.attr("id").attr("id").cast<hid_t>();
                isce3::io::IGroup group(id);
                isce3::core::saveCalGrid(group, name, self, epoch, units);
            },
            "Serialize LUT2d to h5py.Group object (axes assumed range/time).",
            py::arg("h5py_group"),
            py::arg("dataset_name"),
            py::arg("epoch"),
            py::arg("units") = "")

        .def("contains", &LUT2d<T>::contains,
            "Check if point (y, x) resides in domain of LUT",
            py::arg("y"),
            py::arg("x"))
        .def_property_readonly("have_data", &LUT2d<T>::haveData)
        .def_property_readonly("ref_value", &LUT2d<T>::refValue)
        .def_property_readonly("x_start",   &LUT2d<T>::xStart)
        .def_property_readonly("y_start",   &LUT2d<T>::yStart)
        .def_property_readonly("x_spacing", &LUT2d<T>::xSpacing)
        .def_property_readonly("y_spacing", &LUT2d<T>::ySpacing)
        .def_property_readonly("x_axis", &LUT2d<T>::xAxis)
        .def_property_readonly("y_axis", &LUT2d<T>::yAxis)
        .def_property_readonly("length",    &LUT2d<T>::length)
        .def_property_readonly("width",     &LUT2d<T>::width)
        .def_property_readonly("interp_method",
            py::overload_cast<>(&LUT2d<T>::interpMethod, py::const_))
        .def_property("bounds_error",
            py::overload_cast<>(&LUT2d<T>::boundsError, py::const_),
            py::overload_cast<bool>(&LUT2d<T>::boundsError))
        .def_property_readonly("data", [](const LUT2d<T>& self) {
            return self.data().map();
        })
        .def("eval", py::overload_cast<const double, const double>(&LUT2d<T>::eval, py::const_))
        .def("eval", py::overload_cast<double,const Eigen::Ref<const Eigen::VectorXd>&>(&LUT2d<T>::eval, py::const_))
        ;
}

template void addbinding(py::class_<LUT2d<double>> &pyLUT2d);

// end of file
