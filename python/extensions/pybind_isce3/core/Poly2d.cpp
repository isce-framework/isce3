#include "Poly2d.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <vector>

#include <Eigen/Dense>

#include <isce3/core/Poly2d.h>
#include <isce3/except/Error.h>

// Aliases
namespace py = pybind11;
using isce3::core::Poly2d;
using isce3::except::InvalidArgument;
using isce3::except::LengthError;

using RowArrayXXd =
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void addbinding(pybind11::class_<isce3::core::Poly2d>& pyPoly2d)
{
    pyPoly2d.doc() = R"(
        A 2-dimensional power series.

        The Poly2d class represents a polynomial function of range
        (:math:`x`) and azimuth (:math:`y`), defined as follows:

        .. math::

            f\left( y, x \right) = \sum_{i=0}^{N_y} \sum_{j=0}^{N_x} a_{ij}
                \cdot \left( \frac{y-\mu_y}{\sigma_y} \right)^i  \cdot
                \left( \frac{x-\mu_x}{\sigma_x} \right)^j

        Attributes
        ----------
        coeffs : numpy.ndarray
            A 2-D array of polynomial coefficients in order of increasing
            degree, with shape `(y_order+1, x_order+1)`.
        x_mean, y_mean : float
            x/range and y/azimuth offsets.
        x_std, y_std : float
            x/range and y/azimuth scale factors.
        )";

        pyPoly2d
        // constructor
        .def(py::init([](const Eigen::Ref<const RowArrayXXd>& coeffs,
                         double x_mean = 0.0, double y_mean = 0.0,
                         double x_std = 1.0, double y_std = 1.0) {
             if (!(x_std > 0.0))
                 throw InvalidArgument(ISCE_SRCINFO(),
                         "x/range STD must be a positive value");
             if (!(y_std > 0.0))
                 throw InvalidArgument(ISCE_SRCINFO(),
                          "y/azimuth STD must be a positive value");
             auto pf_obj = Poly2d(coeffs.cols() - 1, coeffs.rows() - 1, x_mean,
                     y_mean, x_std, y_std);
             pf_obj.coeffs = std::vector<double>(coeffs.data(),
                     coeffs.data() + coeffs.size());
             return pf_obj;
        }),
        py::arg("coeffs"),
        py::arg("x_mean") = 0.0,
        py::arg("y_mean") = 0.0,
        py::arg("x_std") = 1.0,
        py::arg("y_std") = 1.0)

        // dunder methods
        .def("__repr__",
                [](const Poly2d& self) {
                    std::stringstream os;
                    os << "Poly2d(x_order=" << self.xOrder
                       << ", y_order=" << self.yOrder
                       << ", x_mean=" << self.xMean
                       << ", y_mean=" << self.yMean
                       << ", x_std=" << self.xNorm
                       << ", y_std=" << self.yNorm
                       << ")";
                    return os.str();
                })

        // methods
        .def("eval", &Poly2d::eval, py::arg("y"), py::arg("x"),
                R"(
            Evaluate polynomial at given `y` and `x`.
                )")

        .def("eval",
                [](const Poly2d& self, const Eigen::Ref<const Eigen::ArrayXd>& y,
                   const Eigen::Ref<const Eigen::ArrayXd>& x) {
                    // Check that inputs have compatible shapes.
                    if (x.size() != y.size()) {
                        throw InvalidArgument(ISCE_SRCINFO(),
                                "x & y arrays must have the same shape");
                    }

                    // Evaluate the polynomial at each (x,y) pair.
                    const auto n = x.size();
                    auto z = Eigen::ArrayXd(n);
                    for (Eigen::Index i = 0; i < n; ++i) {
                        z[i] = self.eval(y[i], x[i]);
                    }

                    return z;
                },
                py::arg("y"), py::arg("x"), R"(
                Evaluate the polynomial at the given (y, x) coordinates.

                `y` and `x` must have the same shape.
                )")

        .def("eval",
                [](const Poly2d& self, const Eigen::Ref<const RowArrayXXd>& y,
                   const Eigen::Ref<const RowArrayXXd>& x) {
                    // Check that inputs have compatible shapes.
                    if ((x.rows() != y.rows()) or (x.cols() != y.cols())) {
                        throw InvalidArgument(ISCE_SRCINFO(),
                                "x & y arrays must have the same shape");
                    }

                    // Evaluate the polynomial at each (x,y) pair.
                    const auto m = x.rows();
                    const auto n = x.cols();
                    auto z = RowArrayXXd(m, n);
                    for (Eigen::Index i = 0; i < m; ++i) {
                        for (Eigen::Index j = 0; j < n; ++j) {
                            z(i, j) = self.eval(y(i, j), x(i, j));
                        }
                    }

                    return z;
                },
                py::arg("y"), py::arg("x"), R"(
                Evaluate the polynomial at the given (y, x) coordinates.

                `y` and `x` must have the same shape.
                )")

        .def("evalgrid",
                [](const Poly2d& self,
                   const Eigen::Ref<const Eigen::ArrayXd>& y_vect,
                   const Eigen::Ref<const Eigen::ArrayXd>& x_vect) {
                    // Validate inputs
                    if (x_vect.size() < 1 || y_vect.size() < 1)
                        throw LengthError(ISCE_SRCINFO(),
                                "The size of the vectors must be nonzero");

                    // Evaluate over x and y mesh
                    Eigen::ArrayXXd z(x_vect.size(), y_vect.size());
                    for (Eigen::Index i = 0; i < z.rows(); ++i) {
                        for (Eigen::Index j = 0; j < z.cols(); ++j) {
                            z(i, j) = self.eval(y_vect(j), x_vect(i));
                       }
                    }
                    return z;
                    },
                    py::arg("y_vect"), py::arg("x_vect"),
                    R"(
                Evaluate the polynomial on the Cartesian product of `x_vect` and `y_vect`.
                    )")

        // properties
        .def_property_readonly("x_order",
                [](const Poly2d& self) {return self.xOrder; })
        .def_property_readonly("y_order",
                [](const Poly2d& self) {return self.yOrder; })
        .def_property_readonly("x_mean",
                [](const Poly2d& self) {return self.xMean; })
        .def_property_readonly("y_mean",
                [](const Poly2d& self) {return self.yMean; })
        .def_property_readonly("x_std",
                [](const Poly2d& self) {return self.xNorm; })
        .def_property_readonly("y_std",
                [](const Poly2d& self) {return self.yNorm; })
        .def_property_readonly("coeffs",
                [](const Poly2d& self) {
                    Eigen::Map<const RowArrayXXd> coeffs(
                            self.coeffs.data(), self.yOrder + 1, self.xOrder + 1);
                    return coeffs;
                })
        ;
}
