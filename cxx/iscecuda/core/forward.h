#pragma once

namespace isce { namespace cuda { namespace core {

class gpuBasis;
class Orbit;
class OrbitView;
class ProjectionBase;

// clang-format off
template<class> class gpuInterpolator;
template<class> class gpuLUT1d;
template<class> class gpuLUT2d;
template<class> class gpuSinc2dInterpolator;

template<typename> class BartlettKernel;
template<typename> class LinearKernel;
template<typename> class KnabKernel;
template<typename> class TabulatedKernel;
template<typename> class ChebyKernel;
// clang-format on

}}} // namespace isce::cuda::core
