#include "io.h"
#include "Raster.h"

#include "gdal/gdal.h"

void addsubmodule_io(py::module & m)
{
    py::module m_io = m.def_submodule("io");

    addsubmodule_gdal(m_io);

    py::class_<isce::io::Raster> pyRaster(m_io, "Raster");

    addbinding(pyRaster);
}
