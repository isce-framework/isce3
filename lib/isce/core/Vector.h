#pragma once
#ifndef ISCE_CORE_VECTOR_H
#define ISCE_CORE_VECTOR_H

#include <array>
#include "Common.h"

namespace isce { namespace core {

template<int N>
struct Vector {

private:

    double vdata[N];

public:

    CUDA_HOSTDEV constexpr Vector<N>(const std::array<double, N>& other) {
        #pragma unroll
        for (int i = 0; i < N; i++)
            vdata[i] = other[i];
    }

    template<typename ... Ts>
    CUDA_HOSTDEV constexpr Vector<N>(Ts ... vals) :
        vdata { std::move(vals) ... } {}

    CUDA_HOSTDEV constexpr Vector<N>(const Vector<N>& other) {
        #pragma unroll
        for (int i = 0; i < N; i++)
            vdata[i] = other[i];
    }

    CUDA_HOSTDEV           double& operator[](int i)       { return vdata[i]; }
    CUDA_HOSTDEV constexpr double  operator[](int i) const { return vdata[i]; }

    /*
     * Add two vectors
     */
    CUDA_HOSTDEV constexpr Vector<N> operator+(const Vector<N>& v) const {
        Vector<N> result {};
        #pragma unroll
        for (int i = 0; i < N; i++)
            result[i] = vdata[i] + v[i];
        return result;
    }

    /*
     * Subtrace a vector from another
     */
    CUDA_HOSTDEV constexpr Vector<N> operator-(const Vector<N>& v) const {
        Vector<N> result {};
        #pragma unroll
        for (int i = 0; i < N; i++)
            result[i] = vdata[i] - v[i];
        return result;
    }

    /*
     * Unary minus (vector negation)
     */
    CUDA_HOSTDEV constexpr Vector<N> operator-() const {
        Vector<N> result {};
        #pragma unroll
        for (int i = 0; i < N; i++)
            result[i] = -vdata[i];
        return result;
    }

    /*
     * Multiply a vector by a scalar factor
     */
    CUDA_HOSTDEV constexpr Vector<N> operator*(const double f) const {
        Vector<N> result {};
        #pragma unroll
        for (int i = 0; i < N; i++)
            result[i] = vdata[i] * f;
        return result;
    }
    // Same, but with operands switched
    CUDA_HOSTDEV constexpr friend Vector<N> operator*(const double f, const Vector<N>& v) {
        return v * f;
    }

    /*
     * Divide a vector by a scalar divisor
     */
    CUDA_HOSTDEV constexpr Vector<N> operator/(const double d) const {
        Vector<N> result {};
        #pragma unroll
        for (int i = 0; i < N; i++)
            result[i] = vdata[i] / d;
        return result;
    }

    /*
     *  Calculate the magnitude of vector
     */
    CUDA_HOSTDEV constexpr double norm() const {
        double sum_sq = 0;
        #pragma unroll
        for (int i = 0; i < N; i++)
            sum_sq += vdata[i]*vdata[i];
        return std::sqrt(sum_sq);
    }

    CUDA_HOSTDEV constexpr double dot(const Vector<N>& v) const {
        double result = 0;
        #pragma unroll
        for (int i = 0; i < N; i++)
            result += vdata[i] * v[i];
        return result;
    }

    /*
     *  Calculate the vector cross product against another vector
     */
    CUDA_HOSTDEV constexpr Vector<N> cross(const Vector<N>& v) const;

    /*
     *  Calculate the normalized unit vector
     */
    CUDA_HOSTDEV constexpr Vector<N> unitVec() const {
        return *this / norm();
    }

    /* Pointer interfacing */

    CUDA_HOSTDEV double* data() { return vdata; }
    CUDA_HOSTDEV const double* data() const { return vdata; }

    CUDA_HOST const double* begin() const { return data(); };
    CUDA_HOST const double* end()   const { return data() + N; };
};

template<>
CUDA_HOSTDEV constexpr Vector<3> Vector<3>::cross(const Vector<3>& v) const {
    const Vector<3>& x = *this;
    Vector<3> w {
        x[1] * v[2] - x[2] * v[1],
        x[2] * v[0] - x[0] * v[2],
        x[0] * v[1] - x[1] * v[0]
    };
    return w;
}

using Vec3 = Vector<3>;

// Function to compute normal vector to a plane given three points
CUDA_HOSTDEV inline Vec3 normalPlane(const Vec3& p1,
                                     const Vec3& p2,
                                     const Vec3& p3) {
    const Vec3 p13 = p3 - p1;
    const Vec3 p12 = p2 - p1;
    return p13.cross(p12).unitVec();
}

}} // namespace isce::core

#endif
