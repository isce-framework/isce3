#include "product.h"

#include "GeoGridParameters.h"
#include "RadarGridParameters.h"
#include "Swath.h"

namespace py = pybind11;

void addsubmodule_product(py::module & m)
{
    py::module m_product = m.def_submodule("product");

    // forward declare bound classes
    py::class_<isce3::product::GeoGridParameters> pyGeoGridParameters(m_product, "GeoGridParameters");
    py::class_<isce3::product::RadarGridParameters> pyRadarGridParameters(m_product, "RadarGridParameters");
    py::class_<isce3::product::Swath> pySwath(m_product, "Swath");

    // add bindings
    addbinding(pyGeoGridParameters);
    addbinding(pyRadarGridParameters);
    addbinding(pySwath);
    addbinding_bbox_to_geogrid(m_product);
}
