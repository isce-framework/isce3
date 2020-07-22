#include "signal.h"

#include "Covariance.h"

namespace py = pybind11;

void addsubmodule_signal(py::module & m)
{
    py::module m_signal = m.def_submodule("signal");

    // forward declare bound classes
    py::class_<isce3::signal::Covariance<std::complex<float>>> pyCovarianceComplex64(m_signal, "CovarianceComplex64");
    py::class_<isce3::signal::Covariance<std::complex<double>>> pyCovarianceComplex128(m_signal, "CovarianceComplex128");

    // add bindings
    addbinding(pyCovarianceComplex64);
    addbinding(pyCovarianceComplex128);
}
