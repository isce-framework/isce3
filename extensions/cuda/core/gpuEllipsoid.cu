//
// Author: Joshua Cohen, Liang Yu
// Copyright 2017-2018
//

#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include "gpuEllipsoid.h"
#include "gpuLinAlg.h"

using std::vector;
using isce::cuda::core::gpuEllipsoid;
using isce::cuda::core::gpuLinAlg;

__device__ void gpuEllipsoid::lonLatToXyz(double *llh, double *xyz) const {
    double re = rEast(llh[1]);
    xyz[0] = (re + llh[2]) * cos(llh[1]) * cos(llh[0]);
    xyz[1] = (re + llh[2]) * cos(llh[1]) * sin(llh[0]);
    xyz[2] = ((re * (1. - e2)) + llh[2]) * sin(llh[1]);
}

__device__ void gpuEllipsoid::xyzToLatLon(double *xyz, double *llh) const {
    double p = (pow(xyz[0],2) + pow(xyz[1],2)) / pow(a,2);
    double q = ((1. - e2) * pow(xyz[2],2)) / pow(a,2);
    double r = (p + q - pow(e2,2)) / 6.;
    double s = (pow(e2,2) * p * q) / (4. * pow(r,3));
    double t = cbrt(1. + s + sqrt(s * (2. + s)));
    double u = r * (1. + t + (1. / t));
    double rv = sqrt(pow(u,2) + (pow(e2,2) * q));
    double w = (e2 * (u + rv - q)) / (2. * rv);
    double k = sqrt(u + rv + pow(w,2)) - w;
    double d = (k * sqrt(pow(xyz[0],2) + pow(xyz[1],2))) / (k + e2);
    llh[1] = atan2(xyz[2],d);
    llh[0] = atan2(xyz[1],xyz[0]);
    llh[2] = ((k + e2 - 1.) * sqrt(pow(d,2) + pow(xyz[2],2))) / k;
}

__device__ void gpuEllipsoid::TCNbasis(double *pos, double *vel, double *t, double *c, double *n) {
    double temp[3];
    xyzToLatLon(pos,temp);
    n[0] = -cos(temp[0]) * cos(temp[1]);
    n[1] = -cos(temp[0]) * sin(temp[1]);
    n[2] = -sin(temp[0]);
    gpuLinAlg::cross(n,vel,temp);
    gpuLinAlg::unitVec(temp,c);
    gpuLinAlg::cross(c,n,temp);
    gpuLinAlg::unitVec(temp,t);
}

__global__ void lonLatToXyz_d(gpuEllipsoid elp, double *llh, double *xyz) {
    /*
     *  GPU-side helper kernel for lonLatToXyz_h to use as a consistency check. Note that elp, llh,
     *  and xyz are GPU-side memory constructs.
     */
    elp.lonLatToXyz(llh, xyz);
}

__host__ void gpuEllipsoid::lonLatToXyz_h(cartesian_t &llh, cartesian_t &xyz) {
    /*
     *  CPU-side function to call the corresponding GPU function on a single thread. This function
     *  is primarily meant to be used as a consistency check in the test suite, but may be used in
     *  other contexts.
     */
    // Check inputs for valid length
    //checkVecLen(llh,3);
    //checkVecLen(xyz,3);
    // Malloc memory on the GPU and copy the llh inputs over
    double *llh_d, *xyz_d;
    cudaSetDevice(0);
    cudaMalloc((double**)&llh_d, 3*sizeof(double));
    cudaMalloc((double**)&xyz_d, 3*sizeof(double));
    cudaMemcpy(llh_d, llh.data(), 3*sizeof(double), cudaMemcpyHostToDevice);
    // Run the lonLatToXyz function on the gpuEllipsoid object on the GPU
    dim3 grid(1), block(1);
    lonLatToXyz_d <<<grid,block>>>(*this, llh_d, xyz_d);
    // Copy the resulting xyz back to the CPU-side vector
    cudaMemcpy(xyz.data(), xyz_d, 3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(llh_d);
    cudaFree(xyz_d);
}

__global__ void xyzToLatLon_d(gpuEllipsoid elp, double *xyz, double *llh) {
    /*
     * GPU-side helper kernel for xyzToLatLon_h to use as a consistency check. Note that elp, xyz,
     * and llh are GPU-side memory constructs.
     */
    elp.xyzToLatLon(xyz, llh);
}


__host__ void gpuEllipsoid::xyzToLatLon_h(cartesian_t &xyz, cartesian_t &llh) {
    /*
     *  CPU-side function to call the corresponding GPU function on a single thread. This function
     *  is primarily meant to be used as a consistency check in the test suite, but may be used in
     *  other contexts.
     */
     // Check inputs for valid length
     //checkVecLen(xyz,3);
     //checkVecLen(llh,3);
     // Malloc memory on the GPU and copy the xyz inputs over
     double *xyz_d, *llh_d;
     cudaSetDevice(0);
     cudaMalloc((double**)&xyz_d, 3*sizeof(double));
     cudaMalloc((double**)&llh_d, 3*sizeof(double));
     cudaMemcpy(xyz_d, xyz.data(), 3*sizeof(double), cudaMemcpyHostToDevice);
     // Run the xyzToLatLon function on the gpuEllipsoid object on the GPU
     dim3 grid(1), block(1);
     xyzToLatLon_d <<<grid,block>>>(*this, xyz_d, llh_d);
     // Copy the resulting xyz back to the CPU-side vector
     cudaMemcpy(llh.data(), llh_d, 3*sizeof(double), cudaMemcpyDeviceToHost);
     cudaFree(xyz_d);
     cudaFree(llh_d);
}

