#include "Projections.h"

#include <pybind11/eigen.h>
#include <string>

namespace py = pybind11;

using namespace isce3::core;

void addbinding(py::class_<ProjectionBase, PyProjectionBase>& pyProjectionBase)
{
    pyProjectionBase.doc() = R"(
    Abstract base class for map projections, which handle conversion between
    geodetic and projection coordinates

    Attributes
    ----------
    code : int
        The corresponding EPSG code
    ellipsoid : Ellipsoid
        The geodetic reference ellipsoid
    )";
    pyProjectionBase
            // Constructors
            .def(py::init<int>(), py::arg("code"))

            // Methods
            .def("forward",
                 py::overload_cast<const Vec3&>(&ProjectionBase::forward,
                                                py::const_),
                 py::arg("llh"),
                 R"(
             Forward projection transform from LLH to coordinates in the
             specified projection system.

             Parameters
             ----------
             llh : Vec3
                Longitude (in radians), latitude (in radians), and height above
                ellipsoid (in meters)

             Returns
             -------
             xyz : Vec3
                Coordinates in the specified projection system
             )")
            .def("inverse",
                 py::overload_cast<const Vec3&>(&ProjectionBase::inverse,
                                                py::const_),
                 py::arg("xyz"),
                 R"(
            Inverse projection transform from coordinates in the specified
            projection system to LLH.

            Parameters
            ----------
            xyz : Vec3
                Coordinates in the specified projection system

            Returns
            -------
            llh : Vec3
                Longitude (in radians), latitude (in radians), and height above
                ellipsoid (in meters)
            )")

            // Properties
            .def_property_readonly("code", &ProjectionBase::code)
            .def_property_readonly("ellipsoid", &ProjectionBase::ellipsoid)
            ;
}

void addbinding(py::class_<LonLat>& pyLonLat)
{
    pyLonLat
            .def(py::init<>())
            .def("__repr__", [](const LonLat& self) {
                return "LonLat(epsg=" + std::to_string(self.code()) + ")";
            })
            ;
}

void addbinding(py::class_<Geocent>& pyGeocent)
{
    pyGeocent.doc() = R"(
    Earth-centered, earth-fixed (ECEF) coordinate representation (EPSG: 4978)

    The projection coordinates are ECEF X/Y/Z coordinates, in meters. The WGS 84
    reference ellipsoid is used.
    )";
    pyGeocent
            .def(py::init<>())
            .def("__repr__", [](const Geocent& self) {
                return "Geocent(epsg=" + std::to_string(self.code()) + ")";
            })
            ;
}

void addbinding(py::class_<UTM>& pyUTM)
{
    pyUTM.doc() = R"(
    Universal Transverse Mercator (UTM) projection

    The projection coordinates are easting/northing/height above ellipsoid, in
    meters. The WGS 84 reference ellipsoid is used.

    EPSG codes 32601-32660 correspond to UTM North zones 1-60. EPSG codes
    32701-32760 correspond to UTM South zones 1-60.
    )";
    pyUTM
            .def(py::init<int>(), py::arg("code"))
            .def("__repr__", [](const UTM& self) {
                return "UTM(epsg=" + std::to_string(self.code()) + ")";
            })
            ;
}

void addbinding(py::class_<PolarStereo>& pyPolarStereo)
{
    pyPolarStereo.doc() = R"(
    Universal Polar Stereographic (UPS) projection

    The projection coordinates are easting/northing/height above ellipsoid, in
    meters. The WGS 84 reference ellipsoid is used.

    EPSG code 3413 corresponds to the UPS North zone. EPSG code 3031 corresponds
    to UPS South zone.
    )";
    pyPolarStereo
            .def(py::init<int>(), py::arg("code"))
            .def("__repr__", [](const PolarStereo& self) {
                return "PolarStereo(epsg=" + std::to_string(self.code()) + ")";
            })
            ;
}

void addbinding(py::class_<CEA>& pyCEA)
{
    pyCEA.doc() = R"(
    Cylindrical Equal-Area projection used by the EASE-Grid 2.0 (EPSG: 6933)

    The projection coordinates are easting/northing/height above ellipsoid, in
    meters. The WGS 84 reference ellipsoid is used.
    )";
    pyCEA
            .def(py::init<>())
            .def("__repr__", [](const CEA& self) {
                return "CEA(epsg=" + std::to_string(self.code()) + ")";
            })
            ;
}

void addbinding_makeprojection(pybind11::module& m)
{
    m.def("make_projection", &makeProjection, py::arg("epsg"),
          "Return the projection corresponding to the specified EPSG code.");
}
