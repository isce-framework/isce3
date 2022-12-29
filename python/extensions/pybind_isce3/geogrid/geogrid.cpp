#include "geogrid.h"
#include "getRadarGrid.h"
#include "relocateRaster.h"

void addsubmodule_geogrid(py::module & m)
{
    py::module m_geogrid = m.def_submodule("geogrid");

    addbinding_get_radar_grid(m_geogrid);
    addbinding_relocate_raster(m_geogrid);
}
