#pragma once

#include <pybind11/pybind11.h>

namespace pybind11 {

// Read-only array pseudo-constructor
// Buffer info's read-only flag is currently ignored by pybind11's
// array constructor, so this is the only way to remove the writable flag.
// FIXME: Contribute better readonly support to pybind11!
inline auto roarray(const buffer_info& info, handle base)
{
    array a{pybind11::dtype(info), info.shape, info.strides, info.ptr, base};
    detail::array_proxy(a.ptr())->flags &=
            ~detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return a;
}

} // namespace pybind11
