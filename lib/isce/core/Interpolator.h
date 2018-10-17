//
// Author: Joshua Cohen, Bryan Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_INTERPOLATOR_H
#define ISCE_CORE_INTERPOLATOR_H

#include <complex>
#include <valarray>

// isce::core
#include <isce/core/Matrix.h>

// Declaration
namespace isce {
    namespace core {
        class Interpolator;
    }
}
    
// Definition of Interpolator
class isce::core::Interpolator {
    public:
        // Bilinear interpolation
        template <class U>
        static U bilinear(double, double, const Matrix<U> &);
        // Bicubic interpolation 
        template <class U>
        static U bicubic(double, double, const Matrix<U> &);
        // Compute sinc coefficients 
        static void sinc_coef(double, double, int, double, int, std::valarray<double> &); 
        // Sinc evaluation in 1D 
        template <class U, class V>
        static U sinc_eval(const Matrix<U> &, const Matrix <V> &,
                           int, int, double, int);
        // Sinc evaluation in 2D 
        template <class U, class V>
        static U sinc_eval_2d(const Matrix<U> &, const Matrix<V> &,
                              int, int, double, double, int, int);
        // Spline interpolation 
        template <class U>
        static U interp_2d_spline(int, const Matrix<U> &, double, double);
        // Quadratic interpolation
        static double quadInterpolate(const std::valarray<double> &,
                                      const std::valarray<double> &, double);
        // Akima interpolation
        static double akima(double, double, const Matrix<float> &);
};

// Spline utilities
namespace isce {
    namespace core {
        void initSpline(
            const std::valarray<double> &,
            int, std::valarray<double> &,
            std::valarray<double> &
        );
        double spline(
            double,
            const std::valarray<double> &,
            int,
            const std::valarray<double> &
        );
    }
}

#endif

// end of file
