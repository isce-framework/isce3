//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cuda_runtime.h>
#include <math.h>
#include "gpuLinAlg.h"
using isceLib::gpuLinAlg;

__device__ void LinAlg::cross(double *u, double *v, double *w) {
    w[0] = (u[1] * v[2]) - (u[2] * v[1]);
    w[1] = (u[2] * v[0]) - (u[0] * v[2]);
    w[2] = (u[0] * v[1]) - (u[1] * v[0]);
}

__device__ double LinAlg::dot(double *u, double *v) {
    return ((u[0]*v[0]) + (u[1]*v[1]) + (u[2]*v[2]));
}

__device__ void LinAlg::linComb(double a, double *u, double b, double *v, double *w) {
    w[0] = (a * u[0]) + (b * v[0]);
    w[1] = (a * u[1]) + (b * v[1]);
    w[2] = (a * u[2]) + (b * v[2]);
}

__device__ void LinAlg::unitVec(double *v, double *vhat) {
    double mag = norm(3,v);
    vhat[0] = v[0] / mag;
    vhat[1] = v[1] / mag;
    vhat[2] = v[2] / mag;
}

