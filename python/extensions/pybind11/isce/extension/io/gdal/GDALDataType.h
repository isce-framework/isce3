#pragma once

#include <gdal_priv.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <isce/except/Error.h>

namespace py = pybind11;

namespace isce { namespace extension { namespace io { namespace gdal {

void addbinding(py::enum_<GDALDataType> &);

inline
GDALDataType toGDALDataType(const py::dtype & dt)
{
    if (dt.kind() == '?')                        { return GDT_Byte; }
    if (dt.kind() == 'B')                        { return GDT_Byte; }
    if (dt.kind() == 'u' && dt.itemsize() == 2)  { return GDT_UInt16; }
    if (dt.kind() == 'i' && dt.itemsize() == 2)  { return GDT_Int16; }
    if (dt.kind() == 'u' && dt.itemsize() == 4)  { return GDT_UInt32; }
    if (dt.kind() == 'i' && dt.itemsize() == 4)  { return GDT_Int32; }
    if (dt.kind() == 'f' && dt.itemsize() == 4)  { return GDT_Float32; }
    if (dt.kind() == 'f' && dt.itemsize() == 8)  { return GDT_Float64; }
    if (dt.kind() == 'c' && dt.itemsize() == 8)  { return GDT_CFloat32; }
    if (dt.kind() == 'c' && dt.itemsize() == 16) { return GDT_CFloat64; }

    throw isce::except::RuntimeError(ISCE_SRCINFO(), "dtype is not mappable to GDALDataType");
}

inline
GDALDataType toGDALDataType(const py::object & dt)
{
    return toGDALDataType(py::dtype::from_args(dt));
}

}}}}
