//
// Author: Joshua Cohen, Bryan Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_INTERPOLATOR_H
#define ISCE_CORE_INTERPOLATOR_H

#include <complex>
#include <valarray>

// Declaration
namespace isce {
    namespace core {
        class Interpolator;
        template <class U> class Matrix;
    }
}

// Simple convenience class for 2-dimensional matrices; will replace with
// dedicated array class in the near future
template <class U>
class isce::core::Matrix {
    public:
        // Constructors
        Matrix() : Matrix(1, 1) {};
        Matrix(size_t length, size_t width) : _length(length), _width(width),
            _data(length*width) {}
        // Resize
        void resize(size_t length, size_t width) {
            _length = length;
            _width = width;
            _data.resize(length * width);
        }
        // Access element in 2D
        U & operator()(size_t row, size_t column) {return _data[row*_width + column];}
        // Read-only access element in 2D
        const U & operator()(size_t row, size_t column) const {
            return _data[row*_width + column];
        }
        // Access flattened index
        U & operator()(size_t index) {return _data[index];}
        // Access flattened index in read-only
        const U & operator()(size_t index) const {return _data[index];}
        // Get reference to data
        std::valarray<U> & data() {return _data;}
        // Get geometry
        size_t width() const {return _width;}
        size_t length() const {return _length;}

    private:
        size_t _length;
        size_t _width;
        std::valarray<U> _data;
};
    
// Definition of Interpolator
class isce::core::Interpolator {
    public:
        // Bilinear interpolation
        template <class U>
        static U bilinear(double, double, const Matrix<U> &, int);
        // Bicubic interpolation 
        template <class U>
        static U bicubic(double, double, const Matrix<U> &, int);
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
        static U interp_2d_spline(int, int, int, const Matrix<U> &, double, double);
        // Quadratic interpolation
        static double quadInterpolate(const std::valarray<double> &,
                                      const std::valarray<double> &, double);
        // Akima interpolation
        static double akima(int, int, const Matrix<float> &, double, double);
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
