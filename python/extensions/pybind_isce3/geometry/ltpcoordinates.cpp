#include "ltpcoordinates.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <isce3/geometry/geometry.h>

namespace py = pybind11;
namespace geom = isce3::geometry;

void addbinding_ltp_coordinates(py::module& m)
{

    m.def("heading", &geom::heading, py::arg("lon"), py::arg("lat"),
            py::arg("vel"), R"(
    Get spacecraft heading/track angle from velocity vector at 
    a certain geodetic location of Spacecraft.

    Parameters
    ----------
    lon : float 
        geodetic longitude in radians.
    lat : float
        geodetic latitude in radians.
    vel : isce3.core.Vec3 
        velocity vector or its unit vector in ECEF (x,y,z).

    Returns
    -------
    float
        heading/track angle of spacecraft defined wrt North direction
        in clockwise direction in radians.
)");

    m.def("ned_vector", &geom::nedVector, py::arg("lon"), py::arg("lat"),
            py::arg("vector"), R"(
    Get unit NED(north,east,down) velocity or unit vector from ECEF
    velocity or unit vector at a certain geodetic location of spacecraft.

    Parameters
    ----------
    lon : float     
        geodetic longitude in radians
    lat : float    
        geodetic latitude in radians
    vector : isce3.core.Vec3
        3-D  unit vector or velocity vector in ECEF (x,y,z).

    Returns
    -------
    isce3.core.Vec3
        NED 3-D vector.

    See Also
    --------
    nwu_vector : for NWU local tangent plane coordinate
    enu_vector : for ENU local tangent plane coordinate

    Notes
    -----
    For equations, see [1]_

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates        
)");

    m.def("nwu_vector", &geom::nwuVector, py::arg("lon"), py::arg("lat"),
            py::arg("vector"), R"(
    Get unit NWU(north,west,down) velocity or unit vector from ECEF
    velocity or unit vector at a certain geodetic location of spacecraft.

    Parameters
    ----------
    lon : float     
        geodetic longitude in radians
    lat : float    
        geodetic latitude in radians
    vector : isce3.core.Vec3
        3-D  unit vector or velocity vector in ECEF (x,y,z).

    Returns
    -------
    isce3.core.Vec3
        NWU 3-D vector.

    See Also
    --------
    ned_vector : for NED local tangent plane coordinate
    enu_vector : for ENU local tangent plane coordinate

    Notes
    -----
    For equations, see [1]_

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates 
   
)");

    m.def("enu_vector", &geom::enuVector, py::arg("lon"), py::arg("lat"),
            py::arg("vector"), R"(
    Get unit ENU(north,west,down) velocity or unit vector from ECEF
    velocity or unit vector at a certain geodetic location of spacecraft.

    Parameters
    ----------
    lon : float     
        geodetic longitude in radians
    lat : float    
        geodetic latitude in radians
    vector : isce3.core.Vec3
        3-D  unit vector or velocity vector in ECEF (x,y,z).

    Returns
    -------
    isce3.core.Vec3
        ENU 3-D vector.

    See Also
    --------
    ned_vector : for NED local tangent plane coordinate
    nwu_vector : for NWU local tangent plane coordinate

    Notes
    -----
    For equations, see [1]_

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates 
    
)");
}
