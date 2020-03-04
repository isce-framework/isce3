#include "LUT2d.h"

#include <isce/core/Matrix.h>
#include <isce/except/Error.h>

#include <cstring>
#include <string>
#include <valarray>

#include <pybind11/eigen.h>

namespace py = pybind11;

using isce::core::LUT2d;

auto getInterpMethod = [](const std::string & method)
{
    // check interp name strings to enum interp
    if (method == "sinc")
        return isce::core::SINC_METHOD;
    if (method == "bilinear")
        return isce::core::BILINEAR_METHOD;
    if (method == "bicubic")
        return isce::core::BICUBIC_METHOD;
    if (method == "nearest")
        return isce::core::NEAREST_METHOD;
    if (method == "biquintic")
        return isce::core::BIQUINTIC_METHOD;
    throw isce::except::RuntimeError(ISCE_SRCINFO(), "unrecognized interpolation method");
};

template<typename T>
void addbinding(py::class_<LUT2d<T>> &pyLUT2d)
{

    pyLUT2d
        .def(py::init<>())
        .def(py::init([](double xstart, double ystart,
                    double dx, double dy,
                    py::array_t<T, py::array::c_style | py::array::forcecast> & py_data,
                    std::string &method, bool b_error)
                {
                    // memcpy because ndarray not auto converted to eigen type for some reason
                    if (py_data.ndim() != 2) {
                        throw isce::except::RuntimeError(ISCE_SRCINFO(), "buffer object must be 2-D");
                    }
                    isce::core::Matrix<T> data(py_data.shape()[0], py_data.shape()[1]);
                    std::memcpy(data.data(), py_data.data(), py_data.nbytes());

                    // get interp method
                    auto interp_method = getInterpMethod(method);

                    // return LUT2d object
                    return isce::core::LUT2d<T>(xstart, ystart, dx, dy, data, interp_method, b_error);
                }),
                py::arg("xstart"), py::arg("ystart"), py::arg("dx"), py::arg("dy"),
                py::arg("data"), py::arg("method"), py::arg("b_error")=true)
        .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> & py_xcoord,
                        py::array_t<double, py::array::c_style | py::array::forcecast> & py_ycoord,
                        py::array_t<T, py::array::c_style | py::array::forcecast> & py_data,
                        std::string &method, bool b_error)
                {
                    // memcpy ndarrays because they're not auto converted to eigen type for some reason
                    std::valarray<double> xcoord(py_xcoord.size());
                    std::memcpy(&xcoord[0], py_xcoord.data(), py_xcoord.nbytes());

                    std::valarray<double> ycoord(py_ycoord.size());
                    std::memcpy(&ycoord[0], py_ycoord.data(), py_ycoord.nbytes());

                    if (py_data.ndim() != 2) {
                        throw isce::except::RuntimeError(ISCE_SRCINFO(), "buffer object must be 2-D");
                    }
                    isce::core::Matrix<T> data(py_data.shape()[0], py_data.shape()[1]);
                    std::memcpy(data.data(), py_data.data(), py_data.nbytes());

                    // get interp method
                    auto interp_method = getInterpMethod(method);

                    // return LUT2d object
                    return isce::core::LUT2d<T>(xcoord, ycoord, data, interp_method, b_error);
                }),
                py::arg("xcoord"), py::arg("ycoord"), py::arg("data"), py::arg("method"), py::arg("b_error")=true)
        .def_property_readonly("x_start",   &LUT2d<T>::xStart)
        .def_property_readonly("y_start",   &LUT2d<T>::yStart)
        .def_property_readonly("x_spacing", &LUT2d<T>::xSpacing)
        .def_property_readonly("y_spacing", &LUT2d<T>::ySpacing)
        .def_property_readonly("length",    &LUT2d<T>::length)
        .def_property_readonly("width",     &LUT2d<T>::width)
        .def("eval", &LUT2d<T>::eval)
        ;
}

template void addbinding(py::class_<LUT2d<double>> &pyLUT2d);

// end of file
