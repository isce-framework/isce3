#include "GDALDataType.h"

void addbinding(py::enum_<GDALDataType> & pyGDALDataType)
{
    pyGDALDataType
        .value("GDT_Unknown",  GDT_Unknown)
        .value("GDT_Byte",     GDT_Byte)
        .value("GDT_UInt16",   GDT_UInt16)
        .value("GDT_Int16",    GDT_Int16)
        .value("GDT_UInt32",   GDT_UInt32)
        .value("GDT_Float32",  GDT_Float32)
        .value("GDT_Float64",  GDT_Float64)
        .value("GDT_CInt16",   GDT_CInt16)
        .value("GDT_CInt32",   GDT_CInt32)
        .value("GDT_CFloat32", GDT_CFloat32)
        .value("GDT_CFloat64", GDT_CFloat64)
        .export_values();
}
