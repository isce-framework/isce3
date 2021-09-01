#include "edge_method_cost_func.h"

#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <isce3/antenna/edge_method_cost_func.h>
#include <isce3/core/Linspace.h>
#include <isce3/core/Poly1d.h>

// Aliases
namespace py = pybind11;
namespace ant = isce3::antenna;
using poly1d_t = isce3::core::Poly1d;
using linspace_t = isce3::core::Linspace<double>;

// Functions binding
void addbinding_edge_method_cost_func(py::module& m)
{
    m.def(
            "roll_angle_offset_from_edge",
            [](const poly1d_t& polyfit_echo, const poly1d_t& polyfit_ant,
                    const linspace_t& look_ang,
                    std::optional<poly1d_t> polyfit_weight = {}) {
                return ant::rollAngleOffsetFromEdge(
                        polyfit_echo, polyfit_ant, look_ang, polyfit_weight);
            },
            py::arg("polyfit_echo"), py::arg("polyfit_ant"),
            py::arg("look_ang"), py::arg("polyfit_weight") = std::nullopt,
            R"(
Estimate roll angle offset via edge method from poly-fitted 
power patterns obtained from echo raw data and antenna pattern.
The cost function is solved via Newton method and final solution
is the weighted average of individual solution within look 
(off-nadir) angles [near, far] with desired angle precision all
defined by isce3 Linspace.
See equations for cost function in section 1.1 of the reference [1]_.
The only difference is that the look angles are in (rad) rather than in (deg).
Note that the respective magnitudes for both echo and antenna can be 
either 2-way or 1-way power patterns.

Parameters
----------
polyfit_echo : isce3.core.Poly1d
    to represent polyfitted magnitude of echo (range compressed or raw) data. 
    It must be third-order polynomial of relative magnitude/power in (dB) 
    as a function of look angle in (rad).
polyfit_ant : isce3.core.Poly1d
    to represent polyfitted magnitude of antenna EL power pattern. 
    It must be third-order polynomial of relative magnitude/power in (dB) 
    as a function of look angle in (rad).
    It must have the same mean and std as that of polyfit_echo.    
look_ang : isce3.core.Linspace
    to cover desired range of look angles with desired precision/spacing.
polyfit_weight : isce3.core.Poly1d , optional
    to represent weightings used in final weighted averaged of individual 
    solutions over desired look angle coverage. 
    It shall represent relative magnitude/power in (dB) as a function 
    of look angle in (rad). The order of polynom must be at least 0 (constant weights).

Returns
-------
float
    roll angle offset (rad)
    Note that the roll offset shall be added to EL angles in antenna frame
    to align EL power pattern from antenna to the one extracted from echo given
    the cost function optimized for offset applied to polyfitted antenna data.
float
     max cost function value among all iterations
bool
    overall convergence flag (true or false)
int
    max number of iterations among all iterations

Raises
------
ValueError
    for bad input arguments

Notes
-----
See section 1.1 of references [1]_ for cost function equation.

References
----------
..[1] https://github.jpl.nasa.gov/NISAR-POINTING/DOC/blob/master/El_Pointing_Est_Rising_Edge.pdf

)");

    m.def(
            "roll_angle_offset_from_edge",
            [](const poly1d_t& polyfit_echo, const poly1d_t& polyfit_ant,
                    double look_ang_near, double look_ang_far,
                    double look_ang_prec,
                    std::optional<poly1d_t> polyfit_weight = {}) {
                return ant::rollAngleOffsetFromEdge(polyfit_echo, polyfit_ant,
                        look_ang_near, look_ang_far, look_ang_prec,
                        polyfit_weight);
            },
            py::arg("polyfit_echo"), py::arg("polyfit_ant"),
            py::arg("look_ang_near"), py::arg("look_ang_far"),
            py::arg("look_ang_prec"), py::arg("polyfit_weight") = std::nullopt,
            R"(
Estimate roll angle offset via edge method from poly-fitted 
power patterns obtained from echo raw data and antenna pattern.
The cost function is solved via Newton method and final solution
is the weighted average of individual solution within look 
(off-nadir) angles [near, far] with desired angle precision .
See equations for cost function in section 1.1 of the reference [1]_.
The only difference is that the look angles are in (rad) rather than in (deg).
Note that the respective magnitudes for both echo and antenna can be 
either 2-way or 1-way power patterns.

Parameters
----------
polyfit_echo : isce3.core.Poly1d
    to represent polyfitted magnitude of echo (range compressed or raw) data. 
    It must be third-order polynomial of relative magnitude/power in (dB) 
    as a function of look angle in (rad).
polyfit_ant : isce3.core.Poly1d
    to represent polyfitted magnitude of antenna EL power pattern. 
    It must be third-order polynomial of relative magnitude/power in (dB) 
    as a function of look angle in (rad).
    It must have the same mean and std as that of polyfit_echo.    
look_ang_near : float
    look angle for near range in (rad)
look_ang_far : float
    look angle for far range in (rad)
look_ang_prec : float
    look angle precision/resolution in (rad)
polyfit_weight : isce3.core.Poly1d , optional
    to represent weightings used in final weighted averaged of individual 
    solutions over desired look angle coverage. 
    It shall represent relative magnitude/power in (dB) as a function 
    of look angle in (rad). The order of polynom must be at least 0 (constant weights).

Returns
-------
float
    roll angle offset (rad)
    Note that the roll offset shall be added to EL angles in antenna frame
    to align EL power pattern from antenna to the one extracted from echo given
    the cost function optimized for offset applied to polyfitted antenna data.
float
     max cost function value among all iterations
bool
    overall convergence flag (true or false)
int
    max number of iterations among all iterations

Raises
------
ValueError
    for bad input arguments

Notes
-----
See section 1.1 of references [1]_ for cost function equation.

References
----------
..[1] https://github.jpl.nasa.gov/NISAR-POINTING/DOC/blob/master/El_Pointing_Est_Rising_Edge.pdf

)");
}
