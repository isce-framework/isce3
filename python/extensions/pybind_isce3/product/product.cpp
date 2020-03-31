#include "product.h"

#include "RadarGridParameters.h"
#include "Swath.h"

namespace py = pybind11;

void addsubmodule_product(py::module & m)
{
    py::module m_product = m.def_submodule("product");

    // forward declare bound classes
    py::class_<isce::product::RadarGridParameters> pyRadarGridParameters (m_product, "RadarGridParameters");
    py::class_<isce::product::Swath> pySwath (m_product, "Swath");

    // add bindings
    addbinding(pyRadarGridParameters);
    addbinding(pySwath);
}
