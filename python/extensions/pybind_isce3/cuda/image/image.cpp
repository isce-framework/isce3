#include "image.h"

#include "ResampSlc.h"

namespace py = pybind11;

void addsubmodule_cuda_image(py::module & m)
{
    py::module m_image = m.def_submodule("image");

    // forward declare bound classes
    py::class_<isce3::cuda::image::ResampSlc> pyResampSlc(m_image, "ResampSlc");

    // add bindings
    addbinding(pyResampSlc);
}
