#include "Raster.h"

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <string>

#include <isce3/except/Error.h>
#include <isce3/io/gdal/Buffer.h>

#include <pybind_isce3/roarray.h>

#include "GDALAccess.h"
#include "GDALDataType.h"

using isce3::io::gdal::Buffer;
using isce3::io::gdal::Raster;

template<typename T>
static
py::buffer_info toBuffer(Raster & raster)
{
    Buffer mmap = raster.memmap();

    std::string format = py::format_descriptor<T>::format();
    std::vector<ssize_t> shape = { ssize_t(mmap.length()), ssize_t(mmap.width()) };
    std::vector<ssize_t> strides = { ssize_t(mmap.rowstride()), ssize_t(mmap.colstride()) };
    bool readonly = (mmap.access() == GA_ReadOnly);

    return {mmap.data(), sizeof(T), format, 2, shape, strides, readonly};
}

py::buffer_info toBuffer(Raster & raster)
{
    switch (raster.datatype()) {
        case GDT_Byte       : return toBuffer<unsigned char>(raster);
        case GDT_UInt16     : return toBuffer<std::uint16_t>(raster);
        case GDT_Int16      : return toBuffer<std::int16_t>(raster);
        case GDT_UInt32     : return toBuffer<std::uint32_t>(raster);
        case GDT_Int32      : return toBuffer<std::int32_t>(raster);
        case GDT_Float32    : return toBuffer<float>(raster);
        case GDT_Float64    : return toBuffer<double>(raster);
        case GDT_CFloat32   : return toBuffer<std::complex<float>>(raster);
        case GDT_CFloat64   : return toBuffer<std::complex<double>>(raster);
        default             : break;
    }

    throw isce3::except::RuntimeError(ISCE_SRCINFO(), "unsupported GDAL datatype");
}

template<typename T>
static
Raster toRaster(py::buffer buf)
{
    py::buffer_info info = buf.request();

    if (info.ndim != 2) {
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), "buffer object must be 2-D");
    }

    constexpr static std::size_t max_int = std::numeric_limits<int>::max();
    for (int dim = 0; dim < 2; ++dim) {
        if (info.shape[dim] > max_int) {
            std::string errmsg = "buffer shape along axis " + std::to_string(dim) + " exceeds max size";
            throw isce3::except::OverflowError(ISCE_SRCINFO(), errmsg);
        }
    }

    T * data = static_cast<T *>(info.ptr);
    GDALAccess access = (info.readonly) ? GA_ReadOnly : GA_Update;

    return Raster(data, info.shape[1], info.shape[0], info.strides[1], info.strides[0], access);
}

Raster toRaster(py::buffer buf)
{
    py::buffer_info info = buf.request();
    if (info.format == py::format_descriptor<unsigned char>::format())        { return toRaster<unsigned char>(buf); }
    if (info.format == py::format_descriptor<std::uint16_t>::format())        { return toRaster<std::uint16_t>(buf); }
    if (info.format == py::format_descriptor<std::int16_t>::format())         { return toRaster<std::int16_t>(buf); }
    if (info.format == py::format_descriptor<std::uint32_t>::format())        { return toRaster<std::uint32_t>(buf); }
    if (info.format == py::format_descriptor<std::int32_t>::format())         { return toRaster<std::int32_t>(buf); }
    if (info.format == py::format_descriptor<float>::format())                { return toRaster<float>(buf); }
    if (info.format == py::format_descriptor<double>::format())               { return toRaster<double>(buf); }
    if (info.format == py::format_descriptor<std::complex<float>>::format())  { return toRaster<std::complex<float>>(buf); }
    if (info.format == py::format_descriptor<std::complex<double>>::format()) { return toRaster<std::complex<double>>(buf); }

    throw isce3::except::RuntimeError(ISCE_SRCINFO(), "unable to cast buffer format descriptor to GDALDataType");
}

void addbinding(py::class_<Raster> & pyRaster)
{
    pyRaster
        .def_buffer([](Raster & self) -> py::buffer_info { return toBuffer(self); })
        .def("default_driver", &Raster::defaultDriver, "Default GDAL driver for raster creation")
        .def(py::init<const std::string &, GDALAccess>(),
                "Open an existing file containing a single raster band as a GDAL raster",
                py::arg("path"),
                py::arg("access") = GA_ReadOnly)
        .def(py::init([](const std::string & path, char access)
                {
                    return Raster(path, toGDALAccess(access));
                }),
                "Open an existing file containing a single raster band as a GDAL raster",
                py::arg("path"),
                py::arg("access"))
        .def(py::init<const std::string &, int, GDALAccess>(),
                "Open an existing file as a GDAL dataset and fetch the specified raster band",
                py::arg("path"),
                py::arg("band"),
                py::arg("access") = GA_ReadOnly)
        .def(py::init([](const std::string & path, int band, char access)
                {
                    return Raster(path, band, toGDALAccess(access));
                }),
                "Open an existing file as a GDAL dataset and fetch the specified raster band",
                py::arg("path"),
                py::arg("band"),
                py::arg("access"))
        .def(py::init<const std::string &, int, int, GDALDataType, const std::string &>(),
                "Create a new GDAL dataset containing a single raster band",
                py::arg("path"),
                py::arg("width"),
                py::arg("length"),
                py::arg("datatype"),
                py::arg("driver") = Raster::defaultDriver())
        .def(py::init([](const std::string & path, int width, int length, py::object datatype, const std::string & driver)
                {
                    return Raster(path, width, length, toGDALDataType(datatype), driver);
                }),
                py::arg("path"),
                py::arg("width"),
                py::arg("length"),
                py::arg("datatype"),
                py::arg("driver") = Raster::defaultDriver())
        .def(py::init([](py::buffer buf) { return toRaster(buf); }), py::keep_alive<1, 2>())
        .def("dataset", py::overload_cast<>(&Raster::dataset, py::const_),
                "Get the dataset containing the raster")
        .def_property_readonly("band", &Raster::band, "Band index (1-based)")
        .def_property_readonly("datatype", &Raster::datatype, "Datatype identifier")
        .def_property_readonly("access", &Raster::access, "Access mode")
        .def_property_readonly("width", &Raster::width, "Number of columns")
        .def_property_readonly("length", &Raster::length, "Number of rows")
        .def_property_readonly("driver", &Raster::driver, "Driver name")
        .def_property_readonly("x0", &Raster::x0, "Left edge of left-most pixel in projection coordinates")
        .def_property_readonly("y0", &Raster::y0, "Upper edge of upper-most line in projection coordinates")
        .def_property_readonly("dx", &Raster::dx, "Pixel width in projection coordinates")
        .def_property_readonly("dy", &Raster::dy, "Line height in projection coordinates")
        .def_property_readonly("data", [](Raster & self) { return py::roarray(toBuffer(self), py::cast(self)); })
        ;
}
