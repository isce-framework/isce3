#include "geometryfunc.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <isce3/antenna/Frame.h>
#include <isce3/antenna/geometryfunc.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Quaternion.h>
#include <isce3/core/Vector.h>
#include <isce3/geometry/DEMInterpolator.h>

// Aliases
namespace py = pybind11;
namespace ant = isce3::antenna;
using namespace isce3::core;
using namespace isce3::geometry;

// Functions binding
void addbinding_geometryfunc(py::module& m)
{

    m.def(
            "ant2rgdop",
            [](double el_theta, double az_phi, const Vec3& pos_ecef,
                    const Vec3& vel_ecef, const Quaternion& quat,
                    double wavelength, const DEMInterpolator& dem_interp = {},
                    double abs_tol = 0.5, int max_iter = 10,
                    const ant::Frame& frame = {},
                    const Ellipsoid& ellips = {}) {
                // (slantrange, doppler, convergence)
                return ant::ant2rgdop(el_theta, az_phi, pos_ecef, vel_ecef,
                        quat, wavelength, dem_interp, abs_tol, max_iter, frame,
                        ellips);
            },
            py::arg("el_theta"), py::arg("az_phi"), py::arg("pos_ecef"),
            py::arg("vel_ecef"), py::arg("quaternion"), py::arg("wavelength"),
            py::arg_v("dem_interp", DEMInterpolator(), "0.0"),
            py::arg("abs_tol") = 0.5, py::arg("max_iter") = 10,
            py::arg_v("frame", ant::Frame(), "EL_AND_AZ"),
            py::arg_v("ellips", Ellipsoid(), "WGS84"),
            R"(
Estimate Radar products, Slant range and Doppler centroid, from
spherical angles in antenna body-fixed domain for a certain spacecraft
position, velocity,and attitude at a certain height w.r.t. an ellipsoid.

Parameters
----------
el_theta : float 
    either elevation or theta angle in radians
    depending on the 'frame' object.
az_phi : float 
    either azimuth or phi angle in radians depending
    on the 'frame' object.
pos_ecef : isce3.core.Vec3 
    antenna/spacecraft position in ECEF (m,m,m).
vel_ecef : isce3.core.Vec3
    spacecraft velocity in ECEF (m/s,m/s,m/s).
quaternion : isce3.core.Quaternion 
    quaternion object for transformation from antenna
    body-fixed to ECEF.
wavelength : float 
    Radar wavelength in (m).
dem_interp : isce3.geometry.DEMInterpolator, default=0.0
    isce3 DEMInterpolator object. 
abs_tol : float, default=0.5
    Abs error/tolerance in height estimation (m) between desired 
    input height and final output height. 
max_iter : int, default=10 
    Max number of iterations in height estimation.
frame : isce3.antenna.Frame, default=EL_AND_AZ 
   isce3 Frame object to define antenna spherical coordinate system. 
ellips : isce3.core.Ellipsoid, default=WGS84 
   isce3 Ellipsoid object defining the ellipsoidal planet. 

Returns
-------
float  
    slant range in (m). 
float 
    Doppler centroid in (Hz).
bool
    convergence, true if height tolerance is met,false otherwise.

Raises
------
InvalidArgument
    for bad input argument
RuntimeError
    for non-positive slant range

Notes
-----
See reference [1]_ for algorithm and equations

References
----------
.. [1] https://github.jpl.nasa.gov/SALSA-REE/REE_DOC/blob/master/REE_TECHNICAL_DESCRIPTION.pdf
)");

    m.def(
            "ant2rgdop",
            [](const Eigen::Ref<const Eigen::VectorXd>& el_theta, double az_phi,
                    const Vec3& pos_ecef, const Vec3& vel_ecef,
                    const Quaternion& quat, double wavelength,
                    const DEMInterpolator& dem_interp = {},
                    double abs_tol = 0.5, int max_iter = 10,
                    const ant::Frame& frame = {},
                    const Ellipsoid& ellips = {}) {
                return ant::ant2rgdop(el_theta, az_phi, pos_ecef, vel_ecef,
                        quat, wavelength, dem_interp, abs_tol, max_iter, frame,
                        ellips);
            },
            py::arg("el_theta"), py::arg("az_phi"), py::arg("pos_ecef"),
            py::arg("vel_ecef"), py::arg("quaternion"), py::arg("wavelength"),
            py::arg_v("dem_interp", DEMInterpolator(), "0.0"),
            py::arg("abs_tol") = 0.5, py::arg("max_iter") = 10,
            py::arg_v("frame", ant::Frame(), "EL_AND_AZ"),
            py::arg_v("ellips", Ellipsoid(), "WGS84"),
            R"(
Estimate Radar products, Slant range and Doppler centroid, from
spherical angles in antenna body-fixed domain for a certain spacecraft
position, velocity,and attitude at a certain height w.r.t. an ellipsoid.
Parameters
----------
el_theta : list(float)
    a list of either elevation or theta angles in radians
    depending on the 'frame' object.
az_phi : float 
    either azimuth or phi angle in radians depending
    on the 'frame' object.
pos_ecef : isce3.core.Vec3 
    antenna/spacecraft position in ECEF (m,m,m).
vel_ecef : isce3.core.Vec3
    spacecraft velocity in ECEF (m/s,m/s,m/s).
quaternion : isce3.core.Quaternion 
    quaternion object for transformation from antenna
    body-fixed to ECEF.
wavelength : float 
    Radar wavelength in (m).
dem_interp : isce3.geometry.DEMInterpolator, default=0.0
    isce3 DEMInterpolator object. 
abs_tol : float, default=0.5
    Abs error/tolerance in height estimation (m) between desired 
    input height and final output height. 
max_iter : int, default=10 
    Max number of iterations in height estimation.
frame : isce3.antenna.Frame, default=EL_AND_AZ 
   isce3 Frame object to define antenna spherical coordinate system. 
ellips : isce3.core.Ellipsoid, default=WGS84 
   isce3 Ellipsoid object defining the ellipsoidal planet. 

Returns
-------
numpy.ndarray(float)  
    Array of slant ranges in (m). 
numpy.ndarray(float)
    Array of Doppler centroids in (Hz).
bool
    convergence, true if all height tolerances is met,false otherwise.

Raises
------
InvalidArgument
    for bad input argument
RuntimeError
    for non-positive slant range

Notes
-----
See reference [1]_ for algorithm and equations

References
----------
.. [1] https://github.jpl.nasa.gov/SALSA-REE/REE_DOC/blob/master/REE_TECHNICAL_DESCRIPTION.pdf

)");

    m.def(
            "ant2geo",
            [](double el_theta, double az_phi, const Vec3& pos_ecef,
                    const Quaternion& quat,
                    const DEMInterpolator& dem_interp = {},
                    double abs_tol = 0.5, int max_iter = 10,
                    const ant::Frame& frame = {},
                    const Ellipsoid& ellips = {}) {
                // (pos_llh, convergence)
                return ant::ant2geo(el_theta, az_phi, pos_ecef, quat,
                        dem_interp, abs_tol, max_iter, frame, ellips);
            },
            py::arg("el_theta"), py::arg("az_phi"), py::arg("pos_ecef"),
            py::arg("quaternion"),
            py::arg_v("dem_interp", DEMInterpolator(), "0.0"),
            py::arg("abs_tol") = 0.5, py::arg("max_iter") = 10,
            py::arg_v("frame", ant::Frame(), "EL_AND_AZ"),
            py::arg_v("ellips", Ellipsoid(), "WGS84"),
            R"(
Estimate geodetic geolocation (longitude, latitude, height) from
spherical angles in antenna body-fixed domain for a certain spacecraft
position and attitude at a certain height w.r.t. an ellipsoid.

Parameters
----------
el_theta : float 
    either elevation or theta angle in radians
    depending on the 'frame' object.
az_phi : float 
    either azimuth or phi angle in radians depending
    on the 'frame' object.
pos_ecef : isce3.core.Vec3 
    antenna/spacecraft position in ECEF (m,m,m).
quaternion : isce3.core.Quaternion 
    quaternion object for transformation from antenna
    body-fixed to ECEF.
dem_interp : isce3.geometry.DEMInterpolator, default=0.0
    isce3 DEMInterpolator object. 
abs_tol : float, default=0.5
    Abs error/tolerance in height estimation (m) between desired 
    input height and final output height. 
max_iter : int, default=10 
    Max number of iterations in height estimation.
frame : isce3.antenna.Frame, default=EL_AND_AZ 
   isce3 Frame object to define antenna spherical coordinate system. 
ellips : isce3.core.Ellipsoid, default=WGS84 
   isce3 Ellipsoid object defining the ellipsoidal planet. 

Returns
-------
numpy.ndarray(float)  
    geodetic (longitude, latitude, height) in (rad,rad,m). 
bool
    convergence, true if height tolerance is met, false otherwise.

Raises
------
InvalidArgument
    for bad input argument
RuntimeError
    for non-positive slant range

Notes
-----
See reference [1]_ for algorithm and equations

References
----------
.. [1] https://github.jpl.nasa.gov/SALSA-REE/REE_DOC/blob/master/REE_TECHNICAL_DESCRIPTION.pdf

)");

    m.def(
            "ant2geo",
            [](const Eigen::Ref<const Eigen::VectorXd>& el_theta, double az_phi,
                    const Vec3& pos_ecef, const Quaternion& quat,
                    const DEMInterpolator& dem_interp = {},
                    double abs_tol = 0.5, int max_iter = 10,
                    const ant::Frame& frame = {},
                    const Ellipsoid& ellips = {}) {
                return ant::ant2geo(el_theta, az_phi, pos_ecef, quat,
                        dem_interp, abs_tol, max_iter, frame, ellips);
            },
            py::arg("el_theta"), py::arg("az_phi"), py::arg("pos_ecef"),
            py::arg("quaternion"),
            py::arg_v("dem_interp", DEMInterpolator(), "0.0"),
            py::arg("abs_tol") = 0.5, py::arg("max_iter") = 10,
            py::arg_v("frame", ant::Frame(), "EL_AND_AZ"),
            py::arg_v("ellips", Ellipsoid(), "WGS84"),
            R"(
Estimate geodetic geolocation (longitude, latitude, height) from
spherical angles in antenna body-fixed domain for a certain spacecraft
position and attitude at a certain height w.r.t. an ellipsoid.

Parameters
----------
el_theta : list(float) 
    a list of either elevation or theta angles in radians
    depending on the 'frame' object.
az_phi : float 
    either azimuth or phi angle in radians depending
    on the 'frame' object.
pos_ecef : isce3.core.Vec3 
    antenna/spacecraft position in ECEF (m,m,m).
quaternion : isce3.core.Quaternion 
    quaternion object for transformation from antenna
    body-fixed to ECEF.
dem_interp : isce3.geometry.DEMInterpolator, default=0.0
    isce3 DEMInterpolator object.
abs_tol : float, default=0.5
    Abs error/tolerance in height estimation (m) between desired 
    input height and final output height. 
max_iter : int, default=10 
    Max number of iterations in height estimation.
frame : isce3.antenna.Frame, default=EL_AND_AZ 
   isce3 Frame object to define antenna spherical coordinate system. 
ellips : isce3.core.Ellipsoid, default=WGS84 
   isce3 Ellipsoid object defining the ellipsoidal planet. 

Returns
-------
list(numpy.ndarray(float))  
    a list of geodetic (longitude, latitude, height) in (rad,rad,m). 
bool
    convergence, true if height tolerance is met, false otherwise.

Raises
------
InvalidArgument
    for bad input argument
RuntimeError
    for non-positive slant range

Notes
-----
See reference [1]_ for algorithm and equations

References
----------
.. [1] https://github.jpl.nasa.gov/SALSA-REE/REE_DOC/blob/master/REE_TECHNICAL_DESCRIPTION.pdf

)");

    m.def("range_az_to_xyz", &ant::rangeAzToXyz, py::arg("slant_range"),
            py::arg("az"), py::arg("pos_ecef"), py::arg("quat"),
            py::arg_v("dem_interp", DEMInterpolator(), "DEMInterpolator(0)"),
            py::arg("el_min") = -M_PI / 4, py::arg("el_max") = M_PI / 4,
            py::arg("el_tol") = 0.0,
            py::arg_v("frame", ant::Frame(), "EL_AND_AZ"), R"(
Compute target position given range and AZ angle by varying EL until height
matches DEM.

Parameters
----------
slant_range : float
    Range to target in m
az : float
    AZ angle in radians
pos_ecef : array_like
    ECEF XYZ position of radar in m
quat : isce3.core.Quaternion
    Orientation of the antenna (RCS to ECEF quaternion)
dem_interp : isce3.geometry.DEMInterpolator, optional
    Digital elevation model in m above ellipsoid
el_min : float, optional
    Lower bound for EL solution in rad (default=-45 deg)
el_max : float, optional
    Upper bound for EL solution in rad (default=+45 deg)
el_tol : float, optional
    Allowable absolute error in EL solution in rad
frame : isce3.antenna.Frame, optional
    Coordinate convention for (EL, AZ) to cartesian transformation

Returns
-------
xyz : array_like
    Target position in ECEF in m
)");
}
