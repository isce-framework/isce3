#include "gpuComplex.h"

using isce::cuda::core::gpuComplex;

__host__ __device__ float abs(gpuComplex<float> x) {
    float v, w, t;
    auto a = fabsf(x.r);
    auto b = fabsf(x.i);
    if (a > b) {
        v = a;
        w = b;
    } else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0f + t * t;
    t = v * sqrtf(t);
    if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
        t = v + w;
    }
    return t;
}

__host__ __device__ double abs(gpuComplex<double> x) {
    double v, w, t;
    auto a = fabs(x.r);
    auto b = fabs(x.i);
    if (a > b) {
        v = a;
        w = b;
    } else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0f + t * t;
    t = v * sqrt(t);
    if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
        t = v + w;
    }
    return t;
}
// forced instantiation
//template float abs(float x);
//template double abs(double x);
