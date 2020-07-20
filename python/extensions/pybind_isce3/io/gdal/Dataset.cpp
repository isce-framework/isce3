#include "Dataset.h"

#include <gdal_priv.h>
#include <gdal_utils.h>
#include <memory>
#include <string>

#include <isce3/except/Error.h>
#include <isce3/io/gdal/Raster.h>

#include "GDALAccess.h"
#include "GDALDataType.h"

using isce::io::gdal::Dataset;

static
std::string getInfo(Dataset & dataset)
{
    GDALDatasetH handle = GDALDataset::ToHandle(dataset.get());

    char * tmp = GDALInfo(handle, nullptr);
    if (!tmp) {
        throw isce::except::GDALError(ISCE_SRCINFO(), "failed to display dataset info");
    }
    std::string info(tmp);
    CPLFree(tmp);

    return info;
}

void addbinding(py::class_<Dataset> & pyDataset)
{
    pyDataset
        .def("default_driver", &Dataset::defaultDriver, "Default GDAL driver for dataset creation")
        .def(py::init<const std::string &, GDALAccess>(),
                "Open an existing file as a GDAL dataset.",
                py::arg("path"),
                py::arg("access") = GA_ReadOnly)
        .def(py::init([](const std::string & path, char access)
                {
                    return std::make_unique<Dataset>(path, toGDALAccess(access));
                }),
                "Open an existing file as a GDAL dataset.",
                py::arg("path"),
                py::arg("access"))
        .def(py::init<const std::string &, int, int, int, GDALDataType, const std::string &>(),
                "Create a new GDAL dataset.",
                py::arg("path"),
                py::arg("width"),
                py::arg("length"),
                py::arg("bands"),
                py::arg("datatype"),
                py::arg("driver") = Dataset::defaultDriver())
        .def(py::init([](const std::string & path, int width, int length, int bands, py::object datatype, const std::string & driver)
                {
                    return std::make_unique<Dataset>(path, width, length, bands, toGDALDataType(datatype), driver);
                }),
                py::arg("path"),
                py::arg("width"),
                py::arg("length"),
                py::arg("bands"),
                py::arg("datatype"),
                py::arg("driver") = Dataset::defaultDriver())
        .def_property_readonly("access", &Dataset::access, "Access mode")
        .def_property_readonly("width", &Dataset::width, "Number of columns")
        .def_property_readonly("length", &Dataset::length, "Number of rows")
        .def_property_readonly("bands", &Dataset::bands, "Number of bands")
        .def_property_readonly("driver", &Dataset::driver, "Driver name")
        .def_property_readonly("x0", &Dataset::x0, "Left edge of left-most pixel in projection coordinates")
        .def_property_readonly("y0", &Dataset::y0, "Upper edge of upper-most line in projection coordinates")
        .def_property_readonly("dx", &Dataset::dx, "Pixel width in projection coordinates")
        .def_property_readonly("dy", &Dataset::dy, "Line height in projection coordinates")
        .def("get_raster", &Dataset::getRaster, "Fetch raster band.")
        .def("__repr__", [](Dataset & self) { return getInfo(self); })
        ;
}
