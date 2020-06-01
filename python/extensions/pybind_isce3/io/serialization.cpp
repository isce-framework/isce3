#include "serialization.h"

#include <isce/io/Serialization.h>

namespace py = pybind11;

void addbinding_serialization(py::module& m)
{
    m.def("get_ref_epoch", [](py::object h5group, const std::string& path) {
        auto id = h5group.attr("id").attr("id").cast<hid_t>();
        isce::io::IGroup group(id);
        return isce::io::getRefEpoch(group, path);
    });
}
