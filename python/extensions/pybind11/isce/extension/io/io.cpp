#include "io.h"

#include "gdal/gdal.h"

namespace isce { namespace extension { namespace io {

void addsubmodule(py::module & m)
{
    py::module m_io = m.def_submodule("io");

    gdal::addsubmodule(m_io);
}

}}}
