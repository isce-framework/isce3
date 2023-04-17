/*
 * @file  float2.h
 * @brief Define operators and functions on float2 (cuComplex) datatype
 *
 */

#ifndef __FLOAT2_H
#define __FLOAT2_H

#include <math.h>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

namespace isce3::matchtemplate::pycuampcor {

struct float2 {
    float x, y;
};

struct float3 {
    float x, y, z;
};

struct double2 {
    double x, y;
};

struct int2 {
    int x, y;
};

inline int2 make_int2(int x, int y) {
    return {x, y};
}

inline float2 make_float2(float x, float y) {
    return {x, y};
}

inline double2 make_double2(double x, double y) {
    return {x, y};
}

inline float3 make_float3(float x, float y, float z) {
    return {x, y, z};
}

inline __host__ __device__ void zero(float2 &a) { a.x = 0.0f; a.y = 0.0f; }

// negative
inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}

// complex conjugate
inline __host__ __device__ float2 conjugate(float2 a)
{
    return make_float2(a.x, -a.y);
}

// addition
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
}

inline double2& operator+=(double2& lhs, const float2& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}

// subtraction
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
}

// multiplication
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x = a.x*b.x - a.y*b.y;
    a.y = a.y*b.x + a.x*b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}
inline __host__ __device__ float2 operator*(float2 a, int b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ void operator*=(float2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}
inline __host__ __device__ float2 complexMul(float2 a, float2 b)
{
    return a*b;
}
inline __host__ __device__ float2 complexMulConj(float2 a, float2 b)
{
    return make_float2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y);
}

inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}

// abs, arg
inline __host__ __device__ float complexAbs(float2 a)
{
    return sqrtf(a.x*a.x+a.y*a.y);
}
inline __host__ __device__ float complexArg(float2 a)
{
    return atan2f(a.y, a.x);
}

// make a complex number from phase
inline __host__ __device__ float2 complexExp(float arg)
{
    return make_float2(cosf(arg), sinf(arg));
}

} // namespace

#endif //__FLOAT2_H
// end of file
