#include "Linspace.h"

#include <isce3/except/Error.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using isce3::core::Linspace;
using isce3::except::InvalidArgument;
using isce3::except::OutOfRange;

template<typename T>
void addbinding(py::class_<Linspace<T>>& pyLinspace)
{
    pyLinspace
        // constructor(s)
        .def(py::init([](T first, T spacing, int size) {

                    if (spacing == static_cast<T>(0)) {
                        throw InvalidArgument(ISCE_SRCINFO(), "spacing must be non-zero");
                    }
                    if (size <= 0) {
                        throw InvalidArgument(ISCE_SRCINFO(), "size must be > 1");
                    }
                    return Linspace<T>(first, spacing, size);
                }),
                py::arg("first"),
                py::arg("spacing"),
                py::arg("size"))

        // magic methods
        .def("__getitem__", [](const Linspace<T>& self, int pos) {

                    // wrap index
                    if (pos < 0) {
                        pos += self.size();
                    }

                    if (pos < 0 or pos >= self.size()) {
                        throw OutOfRange(ISCE_SRCINFO(), "index out of range");
                    }
                    return self[pos];
                })
        .def("__getitem__", [](const Linspace<T>& self, py::slice slice) {

                    ssize_t start, stop, step, len;
                    auto res = slice.compute(self.size(), &start, &stop, &step, &len);
                    if (!res) {
                        throw py::error_already_set();
                    }

                    if (step <= 0) {
                        throw InvalidArgument(ISCE_SRCINFO(),
                                "only positive-stride slices are supported");
                    }

                    auto first = self[start];
                    auto spacing = self.spacing() * step;

                    return Linspace<T>(first, spacing, len);
                })
        .def("__len__", &Linspace<T>::size)
        .def("__array__", [](const Linspace<T>& self,
                             std::optional<py::object> dtype,
                             std::optional<bool> copy) {

                    // copy == False (not None or True)
                    if (!copy.value_or(true)) {
                        throw InvalidArgument(ISCE_SRCINFO(), "Unable to avoid "
                            "copy while creating an array as requested.");
                    }
                    py::array_t<T> arr(self.size());
                    auto a = arr.mutable_unchecked();
                    for (int i = 0; i < self.size(); ++i) { a(i) = self[i]; }

                    using namespace pybind11::literals;
                    return arr.attr("astype")(dtype.value_or(py::dtype::of<T>()), "copy"_a=false);
                },
                py::arg("dtype") = py::none(),
                py::arg_v("copy", std::nullopt, "None"))

        // operators
        .def(py::self == py::self)
        .def(py::self != py::self)

        // member access
        .def_property("first",
                py::overload_cast<>(&Linspace<T>::first, py::const_),
                py::overload_cast<T>(&Linspace<T>::first))
        .def_property("spacing",
                py::overload_cast<>(&Linspace<T>::spacing, py::const_),
                [](Linspace<T>& self, T spacing) {

                    if (spacing == static_cast<T>(0)) {
                        throw InvalidArgument(ISCE_SRCINFO(), "spacing must be non-zero");
                    }
                    self.spacing(spacing);
                })
        .def_property_readonly("last", &Linspace<T>::last)
        .def_property_readonly("size", &Linspace<T>::size)
        .def_property_readonly("dtype", [](const Linspace<T>& /*self*/) {

                    return py::dtype::of<T>();
                })
        .def_property_readonly("shape", [](const Linspace<T>& self) {

                    return std::make_tuple(self.size());
                })

        // methods
        .def("resize", [](Linspace<T>& self, int size) {

                    if (size <= 0) {
                        throw InvalidArgument(ISCE_SRCINFO(), "size must be > 1");
                    }
                    self.resize(size);
                },
                py::arg("size"))
        .def("search", [](const Linspace<T>& self, T x) { return self.search(x); },
                "Return the position where the specified value would be inserted "
                "in the sequence in order to maintain sorted order.")
        ;
}

template void addbinding(py::class_<Linspace<double>>&);
