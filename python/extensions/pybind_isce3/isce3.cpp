#include <pybind11/pybind11.h>

#include <isce3/config.h>

#include "antenna/antenna.h"
#include "container/container.h"
#include "core/core.h"
#include "focus/focus.h"
#include "geocode/geocode.h"
#include "geometry/geometry.h"
#include "geogrid/geogrid.h"
#include "image/image.h"
#include "io/io.h"
#include "matchtemplate/matchtemplate.h"
#include "math/math.h"
#include "polsar/polsar.h"
#include "product/product.h"
#include "signal/signal.h"
#include "unwrap/unwrap.h"

#ifdef ISCE3_CUDA
#include "cuda/cuda.h"
#endif

PYBIND11_MODULE(isce3, m)
{
    m.doc() = "InSAR Scientific Computing Environment (ISCE)";
    m.attr("__version__") = isce3::version_string;

    addsubmodule_core(m);
    addsubmodule_geometry(m);
    addsubmodule_geocode(m);
    addsubmodule_geogrid(m);
    addsubmodule_image(m);
    addsubmodule_io(m);
    addsubmodule_matchtemplate(m);
    addsubmodule_math(m);
    addsubmodule_polsar(m);
    addsubmodule_signal(m);
    addsubmodule_product(m);
    addsubmodule_container(m);
    addsubmodule_focus(m);
    addsubmodule_unwrap(m);
    addsubmodule_antenna(m);

#ifdef ISCE3_CUDA
    addsubmodule_cuda(m);
#endif
}
