#include "core.h"

#include "Attitude.h"
#include "Basis.h"
#include "blockProcessing.h"
#include "Constants.h"
#include "DateTime.h"
#include "Ellipsoid.h"
#include "EulerAngles.h"
#include "Interp1d.h"
#include "Interp2d.h"
#include "Kernels.h"
#include "Linspace.h"
#include "LookSide.h"
#include "LUT1d.h"
#include "LUT2d.h"
#include "avgLUT2dToLUT1d.h"
#include "Orbit.h"
#include "Projections.h"
#include "Quaternion.h"
#include "StateVector.h"
#include "TimeDelta.h"
#include "Poly1d.h"
#include "Poly2d.h"
#include "xyzToEnu.h"

namespace py = pybind11;

void addsubmodule_core(py::module & m)
{
    py::module m_core = m.def_submodule("core");

    // forward declare bound classes
    py::class_<isce3::core::Attitude> pyAttitude(m_core, "Attitude");
    py::class_<isce3::core::Basis> pyBasis(m_core, "Basis");
    py::class_<isce3::core::DateTime> pyDateTime(m_core, "DateTime");
    py::class_<isce3::core::Ellipsoid> pyEllipsoid(m_core, "Ellipsoid");
    py::class_<isce3::core::EulerAngles> pyEulerAngles(m_core, "EulerAngles");
    py::class_<isce3::core::Linspace<double>> pyLinspace(m_core, "Linspace");
    py::class_<isce3::core::LUT1d<double>> pyLUT1d(m_core, "LUT1d");
    py::class_<isce3::core::LUT2d<double>> pyLUT2d(m_core, "LUT2d");
    py::class_<isce3::core::Orbit> pyOrbit(m_core, "Orbit");
    py::class_<isce3::core::Quaternion> pyQuaternion(m_core, "Quaternion");
    py::class_<isce3::core::StateVector> pyStateVector(m_core, "StateVector");
    py::class_<isce3::core::TimeDelta> pyTimeDelta(m_core, "TimeDelta");
    py::class_<isce3::core::Poly1d> pyPoly1d(m_core, "Poly1d");
    py::class_<isce3::core::Poly2d> pyPoly2d(m_core, "Poly2d");

    // Default to double for kernels
    using namespace isce3::core;
    py::class_<Kernel<double>, PyKernel<double>> pyKernel(m_core, "Kernel");
    py::class_<BartlettKernel<double>, Kernel<double>>
        pyBartlettKernel(m_core, "BartlettKernel");
    py::class_<LinearKernel<double>, Kernel<double>>
        pyLinearKernel(m_core, "LinearKernel");
    py::class_<KnabKernel<double>, Kernel<double>>
        pyKnabKernel(m_core, "KnabKernel");
    py::class_<NFFTKernel<double>, Kernel<double>>
        pyNFFTKernel(m_core, "NFFTKernel");
    py::class_<AzimuthKernel<double>, Kernel<double>>
        pyAzimuthKernel(m_core, "AzimuthKernel");
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

    py::class_<ProjectionBase, PyProjectionBase> pyProjectionBase(m_core, "ProjectionBase");
    py::class_<LonLat> pyLonLat(m_core, "LonLat", pyProjectionBase);
    py::class_<Geocent> pyGeocent(m_core, "Geocent", pyProjectionBase);
    py::class_<UTM> pyUTM(m_core, "UTM", pyProjectionBase);
    py::class_<PolarStereo> pyPolarStereo(m_core, "PolarStereo", pyProjectionBase);
    py::class_<CEA> pyCEA(m_core, "CEA", pyProjectionBase);

    // forward declare bound enums
    py::enum_<isce3::core::LookSide> pyLookSide(m_core, "LookSide");
    py::enum_<isce3::core::OrbitInterpMethod> pyOrbitInterpMethod(m_core, "OrbitInterpMethod");

    // add bindings
    add_constants(m_core);
    addbinding(pyAttitude);
    addbinding(pyBasis);
    addbinding_block_processing(m_core);
    addbinding_basis(m_core);
    addbinding(pyDateTime);
    addbinding(pyEllipsoid);
    addbinding(pyEulerAngles);
    addbinding(pyLinspace);
    addbinding(pyLookSide);
    addbinding(pyLUT1d);
    addbinding(pyLUT2d);
    addbinding(pyOrbit);
    addbinding(pyOrbitInterpMethod);
    addbinding(pyQuaternion);
    addbinding(pyStateVector);
    addbinding(pyTimeDelta);
    addbinding(pyPoly1d);
    addbinding(pyPoly2d);

    addbinding(pyKernel);
    addbinding(pyBartlettKernel);
    addbinding(pyLinearKernel);
    addbinding(pyKnabKernel);
    addbinding(pyNFFTKernel);
    addbinding(pyAzimuthKernel);
    addbinding(pyTabulatedKernel);
    addbinding(pyChebyKernel);

    addbinding(pyKernelF32);
    addbinding(pyTabulatedKernelF32);
    addbinding(pyChebyKernelF32);

    addbinding(pyProjectionBase);
    addbinding(pyLonLat);
    addbinding(pyGeocent);
    addbinding(pyUTM);
    addbinding(pyPolarStereo);
    addbinding(pyCEA);

    addbinding_interp1d(m_core);
    addbinding_interp2d(m_core);
    addbinding_avgLUT2dToLUT1d(m_core);
    addbinding_makeprojection(m_core);
    addbinding_xyzToEnu(m_core);
    addbinding_get_block_processing_parameters(m_core);
    addbinding_get_block_processing_parameters_y(m_core);
}
