#include "Ellipsoid.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <isce3/core/forward.h>

namespace py = pybind11;

using isce::core::Ellipsoid;
using isce::core::Vec3;

void addbinding(py::class_<Ellipsoid> & pyEllipsoid)
{
    pyEllipsoid
        .def(py::init<>())
        .def(py::init<double, double>(),
                py::arg("a"),
                py::arg("e2"))
        .def_property("a", (double (Ellipsoid::*)() const) &Ellipsoid::a, (void (Ellipsoid::*)(double)) &Ellipsoid::a)
        .def_property("e2", (double (Ellipsoid::*)() const) &Ellipsoid::e2, (void (Ellipsoid::*)(double)) &Ellipsoid::e2)
        .def_property_readonly("b", &Ellipsoid::b,
                        "Return semi-minor axis")
        .def("r_east",  &Ellipsoid::rEast,
                        "Return local radius in EW direction")
        .def("r_north", &Ellipsoid::rNorth,
                        "Return local radius in NS direction")
        .def("r_dir",   &Ellipsoid::rDir,
                        "Return directional local radius")
        .def("lon_lat_to_xyz",  py::overload_cast<const Vec3&>(&Ellipsoid::lonLatToXyz, py::const_),
                                "Transform Lon/Lat/Hgt to ECEF xyz",
                                py::arg("llh"))
        .def("xyz_to_lon_lat",  py::overload_cast<const Vec3 &>(&Ellipsoid::xyzToLonLat, py::const_),
                                "Transform ECEF xyz to Lon/Lat/Hgt",
                                py::arg("xyz"))
        .def("n_vector",        py::overload_cast<double, double>(&Ellipsoid::nVector, py::const_),
                                "Return normal to the ellipsoid at given lon, lat",
                                py::arg("lon"),
                                py::arg("lat"))
        .def("xyz_on_ellipse",  &Ellipsoid::xyzOnEllipse,
                                "Return ECEF coordinates of Lon/Lat point on ellipse")
        ;
}
