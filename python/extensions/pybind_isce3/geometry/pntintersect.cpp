#include "pntintersect.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Vector.h>
#include <isce3/geometry/geometry.h>

namespace py = pybind11;
namespace geom = isce3::geometry;
using Vec3_t = isce3::core::Vec3;
using namespace pybind11::literals;

void addbinding_pnt_intersect(py::module& m)
{
    m.def("slantrange_from_lookvec", &geom::slantRangeFromLookVec,
            py::arg("pos"), py::arg("lkvec"),
            py::arg_v("ellips", isce3::core::Ellipsoid(), "WGS84"),
            R"(
    Get slant range (m) from platform/antenna position in ECEF (x,y,z)
    to Reference Ellipsoid given unit look vector (pointing) in ECEF.

    Parameters
    ----------
    pos : isce3.core.Vec3
        ECEF (x,y,z) positions of antenna/platform in (m,m,m).
    lkvec : isce3.core.Vec3   
        looking/pointing unit vector in ECEF towards planet from
        Antenna/platform.
    ellips : isce3.core.Ellipsoid, optional=WGS84         

    Returns
    -------
    float
        slant range in (m).

    Raises
    ------   
    ValueError 
        zero look vector input argument
    RuntimeError
        Non-positive slant range

    See Also
    --------
    sr_pos_from_lookvec_dem

    Notes
    -----
    See section 6.1 of reference [1]_

    References
    ----------
    .. [1] https://github.jpl.nasa.gov/SALSA-REE/REE_DOC/blob/master/REE_TECHNICAL_DESCRIPTION.pdf
)");

    m.def(
            "sr_pos_from_lookvec_dem",
            [](const Vec3_t& sc_pos, const Vec3_t& lkvec, double dem_hgt = 0.0,
                    double hgt_err = 0.5, int num_iter = 10,
                    const isce3::core::Ellipsoid& ellips = {}) -> py::dict {
                double sr;
                Vec3_t tg_pos, llh;
                auto iter_info = geom::srPosFromLookVecDem(sr, tg_pos, llh,
                        sc_pos, lkvec, dem_hgt, hgt_err, num_iter, ellips);
                return py::dict("iter_info"_a = iter_info, "slantrange"_a = sr,
                        "pos_xyz"_a = tg_pos, "pos_llh"_a = llh);
            },
            py::arg("sc_pos"), py::arg("lkvec"), py::arg("dem_hgt") = 0.0,
            py::arg("hgt_err") = 0.5, py::arg("num_iter") = 10,
            py::arg_v("ellips", isce3::core::Ellipsoid(), "WGS84"),
            R"(
    Get an approximatre ECEF, LLH position and respective Slant range
    at a certain height above the reference ellipsoid of planet for a
    look vector looking from a certain spacecraft position in ECEF
    towards the planet.

    Parameters
    ----------
    sc_pos : isce3.core.Vec3   
        Spacecraft position in ECEF (x,y,z) all in (m)
    lkvec : isce3.core.Vec3
        look unit vector in ECEF (x,y,z), looking from spacecraft 
        towards the planet.
    dem_hgt : float, optional=0.0 
        Desired DEM height (m) above the reference ellipsoid.
    hgt_err : float, optional=0.5
        Max error in height estimation (m) between desired 
        input height and final output height.
    num_iter : int, optional=10
        Max number of iterations in height estimation.
    ellips : isce3.core.Ellipsoid, optional=WGS84

    Returns
    -------
    dict
        A dict with the following keys
        iter_info : tuple of (int, float) 
            number of iterations and height error in (m).
        slantrange : float   
            slant range in (m). 
        pos_xyz : isce3.core.Vec3     
            target position on Ellipsoid in ECEF(x,y,z)  in (m,m,m).
        pos_llh : isce3.core.Vec3
            target position on Ellipsoid in geodtic (lon,lat,height) 
            in (rad,rad,m).

    Raises
    ------
    ValueError
        Bad Iteration or zero look vector input arguments 
    RuntimeError
        Non-positive slant range

    Notes
    -----
    See section 6.1 of reference [1]_

    References
    ----------
    .. [1] https://github.jpl.nasa.gov/SALSA-REE/REE_DOC/blob/master/REE_TECHNICAL_DESCRIPTION.pdf
)");
}
