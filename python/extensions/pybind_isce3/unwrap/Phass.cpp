#include "Phass.h"
#include <isce3/io/Raster.h>

namespace py = pybind11;

using isce3::io::Raster;
using isce3::unwrap::phass::Phass;

void addbinding(py::class_<Phass> & pyPhass)
{
    pyPhass.doc() = R"(
    class for initializing Phass unwrapping algorithm

    Attributes
    ----------
    correlation_threshold : float
        Correlation threshold increment
    good_correlation : float
        Good correlation threshold
    min_pixels_region : int
        Minimum size of a region to be unwrapped
    )";

    pyPhass
    // Constructor
    .def(py::init([](const double correlation_threshold,
                     const double good_correlation,
                     const int min_pixels_region)
               {
                     Phass phass;
                     phass.correlationThreshold(correlation_threshold);
                     phass.goodCorrelation(good_correlation);
                     phass.minPixelsPerRegion(min_pixels_region);

                     return phass;
                }),
                py::arg("correlation_threshold") = 0.2,
                py::arg("good_correlation") = 0.7,
                py::arg("min_pixels_region") = 200
                )
    .def("unwrap", py::overload_cast<Raster&, Raster&, Raster&, Raster&>(&Phass::unwrap),
                py::arg("phase"),
                py::arg("correlation"),
                py::arg("unw_igram"),
                py::arg("label"),
                py::call_guard<py::gil_scoped_release>(),
                R"(
                Perform phase unwrapping using the Phass algorithm

                Parameters
                ----------
                phase: Raster
                    Input interferometric phase (radians)
                correlation: Raster
                    Input interferometric correlation
                unw_igram: Raster
                    Output unwrapped interferogram
                label: Raster
                    Output connected components
                )")
    .def("unwrap", py::overload_cast<Raster&, Raster&, Raster&, Raster&, Raster&>(&Phass::unwrap),
                py::arg("phase"),
                py::arg("power"),
                py::arg("correlation"),
                py::arg("unw_igram"),
                py::arg("label"),
                py::call_guard<py::gil_scoped_release>(),
                R"(
                Perform phase unwrapping using the Phass algorithm

                Parameters
                ----------
                phase: Raster
                    Input interferometric phase (radians)
                power: Raster
                    Power of reference RSLC
                correlation: Raster
                    Input interferometric correlation
                unw_igram: Raster
                    Output unwrapped interferogram
                label: Raster
                    Output connected components
                )")

    // Properties
    .def_property("correlation_threshold",
             py::overload_cast<>(&Phass::correlationThreshold, py::const_),
             py::overload_cast<double>(&Phass::correlationThreshold))
    .def_property("good_correlation",
             py::overload_cast<>(&Phass::goodCorrelation, py::const_),
             py::overload_cast<double>(&Phass::goodCorrelation))
    .def_property("min_pixels_region",
             py::overload_cast<>(&Phass::minPixelsPerRegion, py::const_),
             py::overload_cast<int>(&Phass::minPixelsPerRegion))
    ;
}
