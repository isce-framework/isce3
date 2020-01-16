#include <pybind11/pybind11.h>

#include "io/io.h"

PYBIND11_MODULE(pybind_isce3, m) {
    m.doc() = "InSAR Scientific Computing Environment (ISCE)";

    addsubmodule_io(m);
}
