#pragma once

#include <pybind11/pybind11.h>

#include <isce3/core/Constants.h>
#include <isce3/except/Error.h>

void add_constants(pybind11::module&);

inline isce3::core::dataInterpMethod duck_method(pybind11::object method)
{
    using isce3::core::dataInterpMethod;
    using isce3::core::parseDataInterpMethod;
    if (pybind11::isinstance<pybind11::str>(method)) {
        return parseDataInterpMethod(pybind11::str(method));
    } else if (pybind11::isinstance<dataInterpMethod>(method)) {
        return method.cast<dataInterpMethod>();
    } else {
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "invalid type for interp method");
    }
}
