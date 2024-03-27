#include "image.h"

#include "Resample.h"
#include "ResampSlc.h"

namespace py = pybind11;

void addsubmodule_image(py::module & m)
{
    py::module m_image = m.def_submodule("image");
    py::module m_image_v2 = m_image.def_submodule("v2");
    py::module m_image_modulate = m_image.def_submodule("modulate");

    // Add the resample v2 functionality to the v2 module.
    addbindings_resamp(m_image_v2);
    addbindings_modulate<isce3::core::LUT2d<double>>(m_image_modulate);
    addbindings_modulate<isce3::core::Poly2d>(m_image_modulate);

    // forward declare bound classes for v1
    py::class_<isce3::image::ResampSlc> pyResampSlc(m_image, "ResampSlc");

    // add bindings for v1
    addbinding(pyResampSlc);
}
