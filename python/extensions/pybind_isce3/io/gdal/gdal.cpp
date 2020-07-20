#include "gdal.h"

#include "Dataset.h"
#include "GDALAccess.h"
#include "GDALDataType.h"
#include "Raster.h"

void addsubmodule_gdal(py::module & m)
{
    py::module m_gdal = m.def_submodule("gdal");

    // forward declare bound enums
    py::enum_<GDALAccess> pyGDALAccess(m_gdal, "GDALAccess");
    py::enum_<GDALDataType> pyGDALDataType(m_gdal, "GDALDataType");

    // forward declare bound classes
    py::class_<isce3::io::gdal::Dataset> pyDataset(m_gdal, "Dataset");
    py::class_<isce3::io::gdal::Raster> pyRaster(m_gdal, "Raster", py::buffer_protocol());

    // add bindings
    addbinding(pyGDALAccess);
    addbinding(pyGDALDataType);
    addbinding(pyDataset);
    addbinding(pyRaster);
}
