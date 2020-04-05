#include "Interp1d.h"
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <isce/core/Interp1d.h>
#include <isce/core/Kernels.h>
#include <isce/except/Error.h>

namespace py = pybind11;

using namespace isce::core;
using isce::except::RuntimeError;

template <typename TK, typename TD>
static TD interp(const Kernel<TK> & kernel, py::buffer_info & info, double t)
{
    TD * data = static_cast<TD *>(info.ptr);
    int stride = info.strides[0] / sizeof(TD);
    return interp1d(kernel, data, info.shape[0], stride, t);
}

template <typename T>
void addbinding_interp1d(py::module & m, const char * name)
{
    m.def(name, [](Kernel<T> & kernel, py::buffer buf, double t) {
        py::buffer_info info = buf.request();
        using CT = std::complex<T>;
        if (info.ndim != 1) {
            throw RuntimeError(ISCE_SRCINFO(), "buffer object must be 1-D");
        }
        if (info.format == py::format_descriptor<float>::format()) {
            return py::cast(interp<T,float>(kernel, info, t));
        }
        else if (info.format == py::format_descriptor<double>::format()) {
            return py::cast(interp<T,double>(kernel, info, t));
        }
        else if (info.format == py::format_descriptor<CT>::format()) {
            // NOTE: not mixing float and double to avoid weird promotion rules
            return py::cast(interp<T,CT>(kernel, info, t));
        }
        throw RuntimeError(ISCE_SRCINFO(), "Unsupported types for interp1d");
    });
}

template void addbinding_interp1d<float>(py::module & m, const char * name);
template void addbinding_interp1d<double>(py::module & m, const char * name);
