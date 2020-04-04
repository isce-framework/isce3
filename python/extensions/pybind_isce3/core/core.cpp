#include "core.h"

#include "Constants.h"
#include "DateTime.h"
#include "Ellipsoid.h"
#include "Kernels.h"
#include "LookSide.h"
#include "LUT1d.h"
#include "LUT2d.h"
#include "Orbit.h"
#include "TimeDelta.h"

namespace py = pybind11;

void addsubmodule_core(py::module & m)
{
    py::module m_core = m.def_submodule("core");

    // forward declare bound classes
    py::class_<isce::core::DateTime> pyDateTime(m_core, "DateTime");
    py::class_<isce::core::Ellipsoid> pyEllipsoid(m_core, "Ellipsoid");
    py::class_<isce::core::LUT1d<double>> pyLUT1d(m_core, "LUT1d");
    py::class_<isce::core::LUT2d<double>> pyLUT2d(m_core, "LUT2d");
    py::class_<isce::core::Orbit> pyOrbit(m_core, "Orbit");
    py::class_<isce::core::TimeDelta> pyTimeDelta(m_core, "TimeDelta");

    // Default to double for kernels
    py::class_<isce::core::Kernel<double>, PyKernel<double>> pyKernel(m_core, "Kernel");
    py::class_<isce::core::BartlettKernel<double>, isce::core::Kernel<double>> pyBartlettKernel(m_core, "BartlettKernel");
    py::class_<isce::core::LinearKernel<double>, isce::core::Kernel<double>> pyLinearKernel(m_core, "LinearKernel");
    py::class_<isce::core::KnabKernel<double>, isce::core::Kernel<double>> pyKnabKernel(m_core, "KnabKernel");
    py::class_<isce::core::NFFTKernel<double>, isce::core::Kernel<double>> pyNFFTKernel(m_core, "NFFTKernel");
    py::class_<isce::core::TabulatedKernel<double>, isce::core::Kernel<double>> pyTabulatedKernel(m_core, "TabulatedKernel");
    py::class_<isce::core::ChebyKernel<double>, isce::core::Kernel<double>> pyChebyKernel(m_core, "ChebyKernel");
    // Need float for stuff like rangecomp.
    py::class_<isce::core::Kernel<float>, PyKernel<float>> pyKernelF32(m_core, "KernelF32");
    py::class_<isce::core::KnabKernel<float>, isce::core::Kernel<float>> pyKnabKernelF32(m_core, "KnabKernelF32");

    // forward declare bound enums
    py::enum_<isce::core::LookSide> pyLookSide(m_core, "LookSide");

    // add bindings
    add_constants(m_core);
    addbinding(pyDateTime);
    addbinding(pyEllipsoid);
    addbinding(pyLookSide);
    addbinding(pyLUT1d);
    addbinding(pyLUT2d);
    addbinding(pyOrbit);
    addbinding(pyTimeDelta);
    addbinding(pyKernel);
    addbinding(pyKernelF32);
    addbinding(pyBartlettKernel);
    addbinding(pyLinearKernel);
    addbinding(pyKnabKernel);
    addbinding(pyKnabKernelF32);
    addbinding(pyNFFTKernel);
    addbinding(pyTabulatedKernel);
    addbinding(pyChebyKernel);
}
