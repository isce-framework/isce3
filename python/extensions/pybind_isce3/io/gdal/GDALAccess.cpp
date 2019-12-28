#include "GDALAccess.h"

void addbinding(py::enum_<GDALAccess> & pyGDALAccess)
{
    pyGDALAccess
        .value("GA_ReadOnly", GA_ReadOnly)
        .value("GA_Update", GA_Update)
        .export_values();
}
