//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include "gpuLinAlg.h"

__device__ void
isce::cuda::core::gpuLinAlg::
cross(const double * u, const double * v, double * w) {
    w[0] = (u[1] * v[2]) - (u[2] * v[1]);
    w[1] = (u[2] * v[0]) - (u[0] * v[2]);
    w[2] = (u[0] * v[1]) - (u[1] * v[0]);
}

__device__ double
isce::cuda::core::gpuLinAlg::
dot(const double * u, const double * v) {
    return (u[0]*v[0]) + (u[1]*v[1]) + (u[2]*v[2]);
}

__device__ void
isce::cuda::core::gpuLinAlg::
linComb(double a, const double * u, double b, const double * v, double * w) {
    w[0] = (a * u[0]) + (b * v[0]);
    w[1] = (a * u[1]) + (b * v[1]);
    w[2] = (a * u[2]) + (b * v[2]);
}

__device__ void
isce::cuda::core::gpuLinAlg::
unitVec(const double * v, double * vhat) {
    double mag = norm(v);
    vhat[0] = v[0] / mag;
    vhat[1] = v[1] / mag;
    vhat[2] = v[2] / mag;
}

__device__ double
isce::cuda::core::gpuLinAlg::
norm(const double * v) {
    return std::sqrt(std::pow(v[0], 2) + std::pow(v[1], 2) + std::pow(v[2], 2));
}

__device__ void
isce::cuda::core::gpuLinAlg::
scale(double * v, double scaleFactor) {
    for (int i = 0; i < 3; ++i) v[i] *= scaleFactor;
}

__device__ void
isce::cuda::core::gpuLinAlg::
matVec(const double * t, const double * v, double * w) {
    for (int i = 0; i < 3; ++i) {
        double sum = 0.0;
        for (int j = 0; j < 3; ++j) {
            sum += t[i*3 + j] * v[j];
        }
        w[i] = sum;
    }
}

__device__ void
isce::cuda::core::gpuLinAlg::
tranMat(const double * a, double * aT) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            aT[i*3+j] = a[j*3+i];
        }
    }
}

__device__ void
isce::cuda::core::gpuLinAlg::
enuBasis(double lat, double lon, double * enumat) {
    enumat[0] = -std::sin(lon);
    enumat[1] = -std::sin(lat)*std::cos(lon);
    enumat[2] = std::cos(lat)*std::cos(lon);
    enumat[3] = std::cos(lon);
    enumat[4] = -std::sin(lat)*std::sin(lon);
    enumat[5] = std::cos(lat)*std::sin(lon);
    enumat[6] = 0.0;
    enumat[7] = std::cos(lat);
    enumat[8] = std::sin(lat);
}

// end of file
