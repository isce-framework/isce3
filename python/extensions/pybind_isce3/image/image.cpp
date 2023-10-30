#include "image.h"

#include "Resample.h"
#include "ResampSlc.h"

namespace py = pybind11;

void addsubmodule_image(py::module & m)
{
    py::module m_image = m.def_submodule("image");
    py::module m_image_v2 = m_image.def_submodule("v2");

    addbindings_resamp(m_image_v2);

    // forward declare bound classes
    py::class_<isce3::image::ResampSlc> pyResampSlc(m_image, "ResampSlc");

    // add bindings
    addbinding(pyResampSlc);
}
