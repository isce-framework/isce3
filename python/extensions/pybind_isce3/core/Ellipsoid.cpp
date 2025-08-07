#include "Ellipsoid.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <isce3/core/forward.h>

namespace py = pybind11;

using isce3::core::Ellipsoid;
using isce3::core::Vec3;

void addbinding(py::class_<Ellipsoid> & pyEllipsoid)
{
    pyEllipsoid
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("a"), py::arg("e2"))
        .def_property("a", (double (Ellipsoid::*)() const) &Ellipsoid::a, (void (Ellipsoid::*)(double)) &Ellipsoid::a)
        .def_property("e2", (double (Ellipsoid::*)() const) &Ellipsoid::e2, (void (Ellipsoid::*)(double)) &Ellipsoid::e2)
        .def_property_readonly("b", &Ellipsoid::b, "Return semi-minor axis")
        .def(
            "r_east",
            &Ellipsoid::rEast,
            R"(
            Return the transverse radius of curvature, perpendicular to the north-south direction.

            See `Prime vertical radius <https://en.wikipedia.org/wiki/Earth_radius#Prime_vertical>`_.
            
            Parameters
            ----------
            lat : float
                Latitude in radians
            )"
        )
        .def(
            "r_north",
            &Ellipsoid::rNorth,
            R"(
            Return local radius in NS direction

            See `Meridional radius <https://en.wikipedia.org/wiki/Earth_radius#Meridional>`_.
            
            Parameters
            ----------
            lat : float
                Latitude in radians
            )"
        )
        .def(
            "r_dir",
            &Ellipsoid::rDir,
            R"(
            Return directional local radius

            See `Directional Radius <https://en.wikipedia.org/wiki/Earth_radius#Directional>`_.
            
            Parameters
            ----------
            hdg : float
                Heading, in radians, measured in clockwise direction from the North direction.
            lat : float
                Latitude in radians
            )"
        )
        .def(
            "lon_lat_to_xyz",
            py::overload_cast<const Vec3&>(&Ellipsoid::lonLatToXyz, py::const_),
            R"(
            Transform Lon/Lat/Hgt in radians/radians/meters to ECEF xyz in meters
            
            Parameters
            ----------
            llh : numpy.ndarray of 3 floats
                Latitude (rad), Longitude (rad), Height (m)
            
            Returns
            -------
            xyz : numpy.ndarray of 3 floats
                ECEF Cartesian coordinates in meters.
            )",
            py::arg("llh")
        )
        .def(
            "xyz_to_lon_lat",
            py::overload_cast<const Vec3 &>(&Ellipsoid::xyzToLonLat, py::const_),
            R"(
            Transform ECEF xyz in meters to Lon/Lat/Hgt in radians/radians/meters
            
            Parameters
            ----------
            xyz : numpy.ndarray of 3 floats
                ECEF Cartesian coordinates in meters.
            
            Returns
            -------
            llh : numpy.ndarray of 3 floats
                Latitude (rad), Longitude (rad), Height (m)
            )",
            py::arg("xyz")
        )
        .def(
            "n_vector",
            py::overload_cast<double, double>(&Ellipsoid::nVector, py::const_),
            R"(
            Return normal to the ellipsoid at Lon/Lat, given in radians

            See `N-vector <https://en.wikipedia.org/wiki/N-vector>`_.
            
            Parameters
            ----------
            lon : float
                Longitude in radians
            lon : float
                Latitude in radians
            
            Returns
            -------
            vec : numpy.ndarray of 3 floats
                Unit vector of normal pointing outwards in ECEF cartesian coordinates
            )",
            py::arg("lon"),
            py::arg("lat")
        )
        .def(
            "xyz_on_ellipse",
            &Ellipsoid::xyzOnEllipse,
            R"(
            Return ECEF coordinates of Lon/Lat point, in radians, on ellipse

            See `parametric representation of ellipsoid <https://en.wikipedia.org/wiki/Ellipsoid#Parametric_representation>`_.
            
            Parameters
            ----------
            lon : float
                Longitude in radians
            lon : float
                Latitude in radians
            
            Returns
            -------
            xyz : numpy.ndarray of 3 floats
                ECEF coordinates of point on ellipse
            )"
        )
        ;
}
