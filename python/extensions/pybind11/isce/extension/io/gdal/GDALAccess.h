#pragma once

#include <gdal_priv.h>
#include <pybind11/pybind11.h>
#include <string>

#include <isce/except/Error.h>

namespace py = pybind11;

namespace isce { namespace extension { namespace io { namespace gdal {

void addbinding(py::enum_<GDALAccess> &);

inline
GDALAccess toGDALAccess(char c)
{
    switch (c) {
        case 'r' : return GA_ReadOnly;
        case 'w' : return GA_Update;
    }

    throw isce::except::RuntimeError(ISCE_SRCINFO(), std::string("unsupported access code '") + c + "'");
}

}}}}
