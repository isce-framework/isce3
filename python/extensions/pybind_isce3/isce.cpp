#include <pybind11/pybind11.h>

#include "container/container.h"
#include "core/core.h"
#include "focus/focus.h"
#include "geocode/geocode.h"
#include "geometry/geometry.h"
#include "image/image.h"
#include "io/io.h"
#include "signal/signal.h"
#include "product/product.h"

#ifdef ISCE3_CUDA
#include "cuda/cuda.h"
#endif

PYBIND11_MODULE(pybind_isce3, m) {
    m.doc() = "InSAR Scientific Computing Environment (ISCE)";

    addsubmodule_core(m);
    addsubmodule_geocode(m);
    addsubmodule_geometry(m);
    addsubmodule_image(m);
    addsubmodule_io(m);
    addsubmodule_signal(m);
    addsubmodule_product(m);
    addsubmodule_container(m);
    addsubmodule_focus(m);

#ifdef ISCE3_CUDA
    addsubmodule_cuda(m);
#endif
}
