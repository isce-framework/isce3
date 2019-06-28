#pragma once

namespace isce {
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
        class Quaternion;
        class StateVector;
        class TimeDelta;

        // templates
        template<class T> T sinc(T);
        template<int> class DenseMatrix;
        template<int> class Vector;
        template<class> class Cube;
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

        // using-declarations
        using Mat3 = DenseMatrix<3>;
        using Vec3 = Vector<3>;
        using cartmat_t   = Mat3;
        using cartesian_t = Vec3;
    }
}
