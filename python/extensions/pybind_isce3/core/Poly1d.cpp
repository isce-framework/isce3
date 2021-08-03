#include "Poly1d.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include <Eigen/Dense>

#include <isce3/core/Poly1d.h>
#include <isce3/except/Error.h>

// alias
namespace py = pybind11;
using isce3::core::Poly1d;
using isce3::except::InvalidArgument;
using isce3::except::LengthError;

void addbinding(pybind11::class_<isce3::core::Poly1d>& pyPoly1d)
{
    pyPoly1d
            // constructors
            .def(py::init([](const std::vector<double>& coeffs,
                                  double mean = 0.0, double std = 1.0) {
                if (!(std > 0.0))
                    throw InvalidArgument(
                            ISCE_SRCINFO(), "STD must be a positive value!");
                auto pf_obj = Poly1d(coeffs.size() - 1, mean, std);
                pf_obj.coeffs = coeffs;
                return pf_obj;
            }),
                    py::arg("coeffs"), py::arg("mean") = 0.0,
                    py::arg("std") = 1.0)

            // dunder methods
            .def("__repr__",
                    [](const Poly1d& self) {
                        std::stringstream os;
                        os << "Poly1d(order= " << self.order
                           << ", mean=" << self.mean << ", std=" << self.norm
                           << ")";
                        return os.str();
                    })
            .def("__str__",
                    [](const Poly1d& self) {
                        Eigen::Map<const Eigen::ArrayXd> coeff(
                                self.coeffs.data(), self.coeffs.size());
                        std::stringstream os;
                        os << "Coeffs in ascending order -> " << coeff;
                        return os.str();
                    })
            .def("__call__", [](const Poly1d& self) { return self.coeffs; })
            .def("__getitem__",
                    [](const Poly1d& self, int pos) {
                        return self.coeffs.at(pos);
                    })
            .def("__getitem__",
                    [](const Poly1d& self, py::slice slice) {
                        size_t start, stop, step, slicelength;
                        if (!slice.compute(self.order + 1, &start, &stop, &step,
                                    &slicelength))
                            throw py::error_already_set();
                        py::list result;
                        for (size_t i = 0; i < slicelength; ++i) {
                            result.append(self.coeffs[start]);
                            start += step;
                        }
                        return result;
                    })
            .def("__len__",
                    [](const Poly1d& self) { return self.coeffs.size(); })

            // methods
            .def("eval", &Poly1d::eval, py::arg("x"))
            .def("eval",
                    [](const Poly1d& self, const std::vector<double>& x) {
                        if (x.size() < 1)
                            throw LengthError(ISCE_SRCINFO(),
                                    "The size of the vector must be non zero!");
                        std::vector<double> y;
                        y.reserve(x.size());
                        for (const auto& element : x)
                            y.push_back(self.eval(element));
                        return y;
                    },
		 py::arg("x"))
            .def("derivative",
                    [](const Poly1d& self) { return self.derivative(); })

            // properties
            .def_property_readonly("order",
                    [](const Poly1d& self) { return self.coeffs.size() - 1; })
            .def_property_readonly(
                    "mean", [](const Poly1d& self) { return self.mean; })
            .def_property_readonly(
                    "std", [](const Poly1d& self) { return self.norm; })
            .def_property_readonly(
                    "coeffs", [](const Poly1d& self) { return self.coeffs; })
            .doc() = R"(
Data structure for representing 1D polynomials in ascending order with centralization and scaling)";
}
