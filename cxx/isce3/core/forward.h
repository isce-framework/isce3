#pragma once

namespace isce3 {
    namespace core {
        // plain classes
        class Attitude;
        class Baseline;
        class Basis;
        class DateTime;
        class Ellipsoid;
        class EulerAngles;
        class Metadata;
        class Orbit;
        class Peg;
        class Pixel;
        class Poly1d;
        class Poly2d;
        class ProjectionBase;
        class Quaternion;
        struct StateVector;
        class TimeDelta;

        // templates
        template<int, typename = double> class DenseMatrix;
        template<int, typename = double> class Vector;

        template<class> class Cube;
        template<typename> class Linspace;
        template<class> class LUT1d;
        template<class> class LUT2d;
        template<class> class Matrix;
        // interpolator classes
        template<class> class Interpolator;
        template<class> class BilinearInterpolator;
        template<class> class BicubicInterpolator;
        template<class> class NearestNeighborInterpolator;
        template<class> class Spline2dInterpolator;
        template<class> class Sinc2dInterpolator;
        // kernel classes
        template<class> class Kernel;
        template<class> class BartlettKernel;
        template<class> class KnabKernel;
        template<class> class LinearKernel;
        template<class> class NFFTKernel;
        template<class> class TabulatedKernel;
        template<class> class ChebyKernel;

        // using-declarations
        using Mat3 = DenseMatrix<3>;
        using Vec3 = Vector<3>;
        using cartmat_t   = Mat3;
        using cartesian_t = Vec3;

        // enum types
        enum class LookSide;
        enum class OrbitInterpMethod;
        enum class OrbitInterpBorderMode;
    }
}
