#include "signal.h"

#include "convolve2D.h"
#include "CrossMultiply.h"
#include "Crossmul.h"
#include "flatten.h"
#include "filter2D.h"

namespace py = pybind11;

void addsubmodule_signal(py::module & m)
{
    py::module m_signal = m.def_submodule("signal");

    // forward declare bound classes
    py::class_<isce3::signal::Crossmul> pyCrossmul(m_signal, "Crossmul");
    py::class_<isce3::signal::CrossMultiply> pyCrossMultiply(m_signal,
                                                             "CrossMultiply");

    // add bindings
    addbinding(pyCrossmul);
    addbinding(pyCrossMultiply);
    addbinding_flatten(m_signal);
    addbinding_filter2D(m_signal);
    addbinding_convolve2D<float>(m_signal);
    addbinding_convolve2D<std::complex<float>>(m_signal);
    addbinding_convolve2D<double>(m_signal);
    addbinding_convolve2D<std::complex<double>>(m_signal);
}
