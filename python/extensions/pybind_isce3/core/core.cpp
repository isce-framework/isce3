#include "core.h"

#include "Constants.h"
#include "DateTime.h"
#include "Ellipsoid.h"
#include "Interp1d.h"
#include "Kernels.h"
#include "Linspace.h"
#include "LookSide.h"
#include "LUT1d.h"
#include "LUT2d.h"
#include "Orbit.h"
#include "Quaternion.h"
#include "TimeDelta.h"

namespace py = pybind11;

void addsubmodule_core(py::module & m)
{
    py::module m_core = m.def_submodule("core");

    // forward declare bound classes
    py::class_<isce::core::DateTime> pyDateTime(m_core, "DateTime");
    py::class_<isce::core::Ellipsoid> pyEllipsoid(m_core, "Ellipsoid");
    py::class_<isce::core::Linspace<double>> pyLinspace(m_core, "Linspace");
    py::class_<isce::core::LUT1d<double>> pyLUT1d(m_core, "LUT1d");
    py::class_<isce::core::LUT2d<double>> pyLUT2d(m_core, "LUT2d");
    py::class_<isce::core::Orbit> pyOrbit(m_core, "Orbit");
    py::class_<isce::core::Quaternion> pyQuaternion(m_core, "Quaternion");
    py::class_<isce::core::TimeDelta> pyTimeDelta(m_core, "TimeDelta");

    // Default to double for kernels
    using namespace isce::core;
    py::class_<Kernel<double>, PyKernel<double>> pyKernel(m_core, "Kernel");
    py::class_<BartlettKernel<double>, Kernel<double>>
        pyBartlettKernel(m_core, "BartlettKernel");
    py::class_<LinearKernel<double>, Kernel<double>>
        pyLinearKernel(m_core, "LinearKernel");
    py::class_<KnabKernel<double>, Kernel<double>>
        pyKnabKernel(m_core, "KnabKernel");
    py::class_<NFFTKernel<double>, Kernel<double>>
        pyNFFTKernel(m_core, "NFFTKernel");
    py::class_<TabulatedKernel<double>, Kernel<double>>
        pyTabulatedKernel(m_core, "TabulatedKernel");
    py::class_<ChebyKernel<double>, Kernel<double>>
        pyChebyKernel(m_core, "ChebyKernel");

    // Need Kernel<float> for stuff like rangecomp.
    // Just provide metakernels, with conversions from Kernel<double>.
    py::class_<Kernel<float>, PyKernel<float>> pyKernelF32(m_core, "KernelF32");
    py::class_<TabulatedKernel<float>, Kernel<float>>
        pyTabulatedKernelF32(m_core, "TabulatedKernelF32");
    py::class_<ChebyKernel<float>, Kernel<float>>
        pyChebyKernelF32(m_core, "ChebyKernelF32");

    // forward declare bound enums
    py::enum_<isce::core::LookSide> pyLookSide(m_core, "LookSide");

    // add bindings
    add_constants(m_core);
    addbinding(pyDateTime);
    addbinding(pyEllipsoid);
    addbinding(pyLinspace);
    addbinding(pyLookSide);
    addbinding(pyLUT1d);
    addbinding(pyLUT2d);
    addbinding(pyOrbit);
    addbinding(pyQuaternion);
    addbinding(pyTimeDelta);

    addbinding(pyKernel);
    addbinding(pyBartlettKernel);
    addbinding(pyLinearKernel);
    addbinding(pyKnabKernel);
    addbinding(pyNFFTKernel);
    addbinding(pyTabulatedKernel);
    addbinding(pyChebyKernel);

    addbinding(pyKernelF32);
    addbinding(pyTabulatedKernelF32);
    addbinding(pyChebyKernelF32);

    addbinding_interp1d(m_core);
}
