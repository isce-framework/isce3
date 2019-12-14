#include <pybind11/pybind11.h>

#include "io/io.h"

namespace isce { namespace extension {

PYBIND11_MODULE(isce, m) {
    m.doc() = "InSAR Scientific Computing Environment (ISCE)";

    io::addsubmodule(m);
}

}}
