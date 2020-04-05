#include "Interp1d.h"
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <valarray>
#include <isce/core/Interp1d.h>
#include <isce/core/Kernels.h>
#include <isce/except/Error.h>

namespace py = pybind11;

using namespace isce::core;
using isce::except::RuntimeError;

template <typename TK, typename TD>
static py::object
interp(const Kernel<TK> & kernel, py::buffer_info & info, py::object t)
{
    TD * data = static_cast<TD *>(info.ptr);
    int stride = info.strides[0] / sizeof(TD);
    auto n = info.shape[0];
    if (py::isinstance<py::float_>(t)) {
        return py::cast(interp1d(kernel, data, n, stride, py::float_(t)));
    }
    else if (py::isinstance<py::array_t<double>>(t)) {
        auto ta = py::array_t<double>(t).unchecked<1>();
        std::valarray<TD> out(ta.size());
        #pragma omp parallel for
        for (size_t i=0; i < ta.size(); ++i) {
            out[i] = interp1d(kernel, data, n, stride, ta(i));
        }
        return py::cast(out, py::return_value_policy::take_ownership);
    }
    throw RuntimeError(ISCE_SRCINFO(),
        "interp1d expects time is float or numpy.float64 array");
}

template <typename T>
void addbinding_interp1d(py::module & m, const char * name)
{
    m.def(name, [](Kernel<T> & kernel, py::buffer buf, py::object t) {
        py::buffer_info info = buf.request();
        using CT = std::complex<T>;
        if (info.ndim != 1) {
            throw RuntimeError(ISCE_SRCINFO(), "data buffer must be 1-D");
        }
        if (info.format == py::format_descriptor<float>::format()) {
            return interp<T,float>(kernel, info, t);
        }
        else if (info.format == py::format_descriptor<double>::format()) {
            return interp<T,double>(kernel, info, t);
        }
        else if (info.format == py::format_descriptor<CT>::format()) {
            // NOTE: not mixing float and double to avoid weird promotion rules
            return interp<T,CT>(kernel, info, t);
        }
        throw RuntimeError(ISCE_SRCINFO(), "Unsupported types for interp1d");
    });
}

template void addbinding_interp1d<float>(py::module & m, const char * name);
template void addbinding_interp1d<double>(py::module & m, const char * name);
