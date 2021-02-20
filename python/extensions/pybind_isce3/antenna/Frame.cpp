#include "Frame.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <isce3/antenna/SphGridType.h>

namespace py = pybind11;
using isce3::antenna::Frame;
using isce3::antenna::SphGridType;

using dvec_t = std::vector<double>;
using vec3_t = typename Frame::Vec3_t;

void addbinding(py::class_<Frame>& pyFrame)
{

    pyFrame.def(py::init<>())
            .def(py::init<SphGridType>(), py::arg("grid_type"))
            .def(py::init<const std::string&>(), py::arg("grid_str"))

            .def_property_readonly("grid_type",
                    py::overload_cast<>(&Frame::gridType, py::const_))

            .def("__repr__",
                    [](const Frame& f) {
                        return "Frame('" + toStr(f.gridType()) + "')";
                    })

            .def("sph2cart",
                    py::overload_cast<double, double>(
                            &Frame::sphToCart, py::const_),
                    "Spherical to Cartesian Coordinate", py::arg("el_theta"),
                    py::arg("az_phi"))
            .def("sph2cart",
                    py::overload_cast<const dvec_t&, const dvec_t&>(
                            &Frame::sphToCart, py::const_),
                    "Spherical to Cartesian Coordinate", py::arg("el_theta"),
                    py::arg("az_phi"))
            .def("sph2cart",
                    py::overload_cast<const dvec_t&, double>(
                            &Frame::sphToCart, py::const_),
                    "Spherical to Cartesian Coordinate", py::arg("el_theta"),
                    py::arg("az_phi"))
            .def("sph2cart",
                    py::overload_cast<double, const dvec_t&>(
                            &Frame::sphToCart, py::const_),
                    "Spherical to Cartesian Coordinate", py::arg("el_theta"),
                    py::arg("az_phi"))

            .def("cart2sph",
                    py::overload_cast<vec3_t>(&Frame::cartToSph, py::const_),
                    "Cartesian to Spherical Coordinate")
            .def("cart2sph",
                    py::overload_cast<std::vector<vec3_t>>(
                            &Frame::cartToSph, py::const_),
                    "Cartesian to Spherical Coordinate")

            .def(py::self == py::self)
            .def(py::self != py::self)
            .doc() = "A class for antenna frame and spherical-cartesian "
                     "coordinate transformation";
}
