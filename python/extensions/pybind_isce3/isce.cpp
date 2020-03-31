#include <pybind11/pybind11.h>

#include "core/core.h"
#include "io/io.h"
#include "product/product.h"

PYBIND11_MODULE(pybind_isce3, m) {
    m.doc() = "InSAR Scientific Computing Environment (ISCE)";

    addsubmodule_core(m);
    addsubmodule_io(m);
    addsubmodule_product(m);
}
