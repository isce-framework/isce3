#include "Interp2d.h"
#include "Kernels.h"
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <isce3/core/Interp2d.h>
#include <isce3/core/Kernels.h>
#include <isce3/core/Linspace.h>
#include <isce3/except/Error.h>
#include <isce3/math/complexOperations.h>

namespace py = pybind11;

using namespace isce3::core;
using isce3::except::RuntimeError;

template <typename KernelType, typename DataType>
static py::object
interp_duckt(const Kernel<KernelType>& kernelx,
             const Kernel<KernelType>& kernely,
             py::buffer_info & info, py::object t)
{
    if (info.ndim != 2) {
        throw RuntimeError(ISCE_SRCINFO(), "expected data.ndim == 2");
    }
    DataType* data = static_cast<DataType*>(info.ptr);
    int stridex = info.strides[1] / sizeof(DataType);
    int stridey = info.strides[0] / sizeof(DataType);
    auto nx = info.shape[1];
    auto ny = info.shape[0];
    if (py::isinstance<py::tuple>(t)) {
        py::tuple tup(t);
        double x = py::float_(tup[0]), y = py::float_(tup[1]);
        return py::cast(interp2d(kernelx, kernely, data, nx, stridex, ny, stridey, x, y));
    }
    else if (py::isinstance<py::array_t<double>>(t)) {
        auto ta = py::array_t<double>(t).unchecked<2>();
        if (ta.shape(1) != 2) {
            throw RuntimeError(ISCE_SRCINFO(), "expected t.shape[1] == 2");
        }
        const auto nout = ta.shape(0);
        py::array_t<DataType> out(nout);
        auto outbuf = out.mutable_data();
        if (is_cpp_kernel(kernelx) and is_cpp_kernel(kernely)) {
            py::gil_scoped_release release;
            #pragma omp parallel for
            for (size_t i = 0; i < nout; ++i) {
                outbuf[i] = interp2d(kernelx, kernely, data, nx, stridex,
                    ny, stridey, ta(i, 0), ta(i, 1));
            }
        } else {
            // don't release GIL since kernel is a Python object
            for (size_t i = 0; i < nout; ++i) {
                outbuf[i] = interp2d(kernelx, kernely, data, nx, stridex,
                    ny, stridey, ta(i, 0), ta(i, 1));
            }
        }
        return out;
    }
    throw RuntimeError(ISCE_SRCINFO(),
        "interp2d expects time is tuple[float,float] or numpy.float64 array");
}

template <typename T>
static py::object
interp_duckbuf(Kernel<T>& kernelx, Kernel<T>& kernely, py::buffer buf, py::object t)
{
    py::buffer_info info = buf.request();
    using C8 = std::complex<float>;
    using C16 = std::complex<double>;
    if (info.ndim != 2) {
        throw RuntimeError(ISCE_SRCINFO(), "data buffer must be 2D");
    }
    if (info.format == py::format_descriptor<float>::format()) {
        return interp_duckt<T,float>(kernelx, kernely, info, t);
    }
    else if (info.format == py::format_descriptor<double>::format()) {
        return interp_duckt<T,double>(kernelx, kernely, info, t);
    }
    else if (info.format == py::format_descriptor<C8>::format()) {
        return interp_duckt<T,C8>(kernelx, kernely, info, t);
    }
    else if (info.format == py::format_descriptor<C16>::format()) {
        return interp_duckt<T,C16>(kernelx, kernely, info, t);
    }
    throw RuntimeError(ISCE_SRCINFO(), "Unsupported types for interp2d");
}

void addbinding_interp2d(py::module & m)
{
    m.def("interp2d", [](py::object pyKernelX, py::object pyKernelY, py::buffer buf, py::object t) {
        if (py::isinstance<Kernel<float>>(pyKernelX) and py::isinstance<Kernel<float>>(pyKernelY)) {
            auto kernelx = pyKernelX.cast<Kernel<float> *>();
            auto kernely = pyKernelY.cast<Kernel<float> *>();
            return interp_duckbuf(*kernelx, *kernely, buf, t);
        }
        else if (py::isinstance<Kernel<double>>(pyKernelX) and py::isinstance<Kernel<double>>(pyKernelY)) {
            auto kernelx = pyKernelX.cast<Kernel<double> *>();
            auto kernely = pyKernelY.cast<Kernel<double> *>();
            return interp_duckbuf(*kernelx, *kernely, buf, t);
        }
        throw RuntimeError(ISCE_SRCINFO(), "Expected Kernel or KernelF32 (and matching type of X and Y)");
    },
    R"(
    Interpolate a 2D array

    Parameters
    ----------
    kernelx : isce3.core.Kernel
        Interpolation kernel to use in x-direction (across columns)
    kernely : isce3.core.Kernel
        Interpolation kernel to use in y-direction (down rows)
    data : numpy.ndarray
        Two-dimensional array of data to interpolate.  Type must be float32,
        float64, or their complex counterparts.
    time : Union[tuple[float, float], numpy.ndarray]
        Desired interpolation point(s).  Either a tuple of (x, y) or an Nx2
        array of (x, y) points.  The (x, y) values are in pixel units, e.g.,
        the x values should be in the range [0, ncols - 1] and the y values
        should be in the range [0, nrows - 1] where nrows, ncols = data.shape

    Returns
    -------
    out 
        The interpolated data.  If `time` is a tuple then it is a scalar.
        Otherwise it is a 1D array the same length as the `time` argument.
    
    Notes
    -----
    The algorithm zero-fills any required samples out of bounds of the input
    array.  So returned values may be invalid around (n-1)/2 samples from the
    edges where n is the kernel width, and points requested outside that region
    will be zero.
    )",
    py::arg("kernelx"), py::arg("kernely"), py::arg("data"), py::arg("time"));
}
