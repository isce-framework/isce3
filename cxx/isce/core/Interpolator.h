//
// Author: Joshua Cohen, Bryan Riel, Liang Yu
// Copyright 2017-2018
//

#pragma once

#include "forward.h"

#include <valarray>

#include "Constants.h"
#include "EMatrix.h"
#include "Matrix.h"

/** Definition of parent Interpolator */
template<typename U>
class isce::core::Interpolator {

protected:
    using Map = typename Eigen::Map<const EArray2D<U>>;

    // Public interface
public:
    /** Virtual destructor (allow destruction of base Interpolator pointer) */
    virtual ~Interpolator() {}

    /** Interpolate at a given coordinate for an input Eigen::Map */
    virtual U interpolate(double x, double y, const Map& map) const = 0;

    /** Interpolate at a given coordinate for an input isce::core::Matrix */
    U interpolate(double x, double y, const Matrix<U>& z) const
    {
        return interpolate(x, y, z.map());
    }

    /** Interpolate at a given coordinate for data passed as a valarray */
    U interpolate(double x, double y, std::valarray<U>& z_data,
                  size_t width) const
    {
        const Map z {&z_data[0],
                     static_cast<Eigen::Index>(z_data.size() / width),
                     static_cast<Eigen::Index>(width)};
        return interpolate(x, y, z);
    }

    /** Interpolate at a given coordinate for data passed as a vector */
    U interpolate(double x, double y, std::vector<U>& z_data,
                  size_t width) const
    {
        const Map z {&z_data[0],
                     static_cast<Eigen::Index>(z_data.size() / width),
                     static_cast<Eigen::Index>(width)};
        return interpolate(x, y, z);
    }

    /** Return interpolation method. */
    dataInterpMethod method() const { return _method; }

    // Protected constructor and data to be used by derived classes
protected:
    inline Interpolator(dataInterpMethod method) : _method {method} {}
    dataInterpMethod _method;
};

/** Definition of BilinearInterpolator */
template<typename U>
class isce::core::BilinearInterpolator : public isce::core::Interpolator<U> {

    using super_t = Interpolator<U>;
    using typename super_t::Map;

public:
    /** Default constructor */
    BilinearInterpolator() : super_t {BILINEAR_METHOD} {}

    /** Interpolate at a given coordinate. */
    U interpolate(double x, double y, const Map& z) const override;

    // Inherit overloads for other datatypes
    using super_t::interpolate;
};

/** Definition of BicubicInterpolator */
template<typename U>
class isce::core::BicubicInterpolator : public isce::core::Interpolator<U> {

    using super_t = Interpolator<U>;
    using typename super_t::Map;

public:
    /** Default constructor */
    BicubicInterpolator() : super_t {BICUBIC_METHOD} {}

    /** Interpolate at a given coordinate. */
    U interpolate(double x, double y, const Map& z) const override;

    // Inherit overloads for other datatypes
    using super_t::interpolate;
};

/** Definition of NearestNeighborInterpolator */
template<typename U>
class isce::core::NearestNeighborInterpolator :
    public isce::core::Interpolator<U> {

    using super_t = Interpolator<U>;
    using typename super_t::Map;

public:
    /** Default constructor */
    NearestNeighborInterpolator() : super_t {NEAREST_METHOD} {}

    /** Interpolate at a given coordinate. */
    U interpolate(double x, double y, const Map& z) const override;

    // Inherit overloads for other datatypes
    using super_t::interpolate;
};

/** Definition of Spline2dInterpolator */
template<typename U>
class isce::core::Spline2dInterpolator : public isce::core::Interpolator<U> {

    using super_t = Interpolator<U>;
    using typename super_t::Map;

public:
    /** Default constructor. */
    Spline2dInterpolator(size_t order);

    /** Interpolate at a given coordinate. */
    U interpolate(double x, double y, const Map& z) const override;

    // Inherit overloads for other datatypes
    using super_t::interpolate;

    // Data members
private:
    size_t _order;

    // Utility spline functions
private:
    void _initSpline(const std::valarray<U>&, int, std::valarray<U>&,
                     std::valarray<U>&) const;

    U _spline(double, const std::valarray<U>&, int,
              const std::valarray<U>&) const;
};

/** Definition of Sinc2dInterpolator */
template<typename U>
class isce::core::Sinc2dInterpolator : public isce::core::Interpolator<U> {

    using super_t = Interpolator<U>;
    using typename super_t::Map;

public:
    /** Default constructor. */
    Sinc2dInterpolator(int sincLen, int sincSub);

    /** Interpolate at a given coordinate. */
    U interpolate(double x, double y, const Map& z) const override;

    // Inherit overloads for other datatypes
    using super_t::interpolate;

private:
    // Compute sinc coefficients
    void _sinc_coef(double beta, double relfiltlen, int decfactor,
                    double pedestal, int weight,
                    std::valarray<double>& filter) const;

    // Evaluate sinc
    U _sinc_eval_2d(const Map& z, int intpx, int intpy, double frpx,
                    double frpy) const;

private:
    Matrix<double> _kernel;
    int _kernelLength, _kernelWidth, _sincHalf;
};

// Extra interpolation and utility functions
namespace isce { namespace core {

/** Utility function to create interpolator pointer given an interpolator enum
 * type */
template<typename U>
inline Interpolator<U>*
createInterpolator(dataInterpMethod method, size_t order = 6,
                   int sincLen = SINC_LEN, int sincSub = SINC_SUB)
{
    if (method == BILINEAR_METHOD) {
        return new BilinearInterpolator<U>();
    } else if (method == BICUBIC_METHOD) {
        return new BicubicInterpolator<U>();
    } else if (method == BIQUINTIC_METHOD) {
        return new Spline2dInterpolator<U>(order);
    } else if (method == NEAREST_METHOD) {
        return new NearestNeighborInterpolator<U>();
    } else if (method == SINC_METHOD) {
        return new Sinc2dInterpolator<U>(sincLen, sincSub);
    } else {
        return new BilinearInterpolator<U>();
    }
}

// Sinc evaluation in 1D
template<class U, class V>
U sinc_eval(const Matrix<U>&, const Matrix<V>&, int, int, double, int);

}} // namespace isce::core
