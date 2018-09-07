//
// Author: Joshua Cohen
// Copyright 2017
//
// NOTE: This class is the most complicated in the CUDA-specific subset of isceLib because we need 
//       to carefully manage the deep-copying in the constructors (so we don't have to worry about 
//       adding it to every code that uses this class)

// Needed to call the cudaMalloc/cudaFree APIs
#include <cuda_runtime.h>
#include <vector>
#include "isce/core/Constants.h"
#include "gpuOrbit.h"

using isce::cuda::core::gpuOrbit;
using isce::core::cartesian_t;
using isce::core::Orbit;
using std::vector;

// Advanced "copy" constructor to handle deep-copying of Orbit data (only callable by host). Owner 
// member variable indicates that only the host-side copy of the gpuOrbit can handle freeing the 
// memory (device-side copy constructor for gpuOrbit sets owner to false)
__host__ gpuOrbit::gpuOrbit(const Orbit &orb) : 
    nVectors(orb.nVectors),
    owner(true) {
    cudaSetDevice(0);
    // Malloc device-side memory (this API is host-side only)
    cudaMalloc((double**)&UTCtime, nVectors*sizeof(double));
    cudaMalloc((double**)&position, 3*nVectors*sizeof(double));
    cudaMalloc((double**)&velocity, 3*nVectors*sizeof(double));
    // Copy Orbit data to device-side memory and keep device pointer in gpuOrbit object. Device-side 
    // copy constructor simply shallow-copies the device pointers when called
    cudaMemcpy(UTCtime, &(orb.UTCtime[0]), nVectors*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(position, &(orb.position[0]), 3*nVectors*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(velocity, &(orb.velocity[0]), 3*nVectors*sizeof(double), cudaMemcpyHostToDevice);
}

// Both the host-side and device-side copies of the gpuOrbit will call the destructor, so we have to 
// implement a way of having an arbitrary copy on host OR device determine when to free the memory 
// (since only the original host-side copy should free)
gpuOrbit::~gpuOrbit() {
    if (owner) {
        cudaFree(UTCtime);
        cudaFree(position);
        cudaFree(velocity);
    }
}

__device__ int gpuOrbit::interpolateWGS84Orbit(double tintp, double *opos, double *ovel) const {
    if (nVectors < 4) return 1;
    if ((tintp < UTCtime[0]) || (tintp > UTCtime[nVectors-1])) return 1;
    int idx = -1;
    for (int i=0; i<nVectors; i++) {
        if ((UTCtime[i] >= tintp) && (idx == -1)) {
            idx = min(max(i-2, 0), nVectors-4);
        }
    }
    
    double f0[4];
    double f1[4] = {tintp - UTCtime[idx], 
                    tintp - UTCtime[idx+1], 
                    tintp - UTCtime[idx+2], 
                    tintp - UTCtime[idx+3]};
    double sum = (1. / (UTCtime[idx] - UTCtime[idx+1])) + (1. / (UTCtime[idx] - UTCtime[idx+2])) + 
                 (1. / (UTCtime[idx] - UTCtime[idx+3]));
    f0[0] = 1. - (2. * (tintp - UTCtime[idx]) * sum);
    sum = (1. / (UTCtime[idx+1] - UTCtime[idx])) + (1. / (UTCtime[idx+1] - UTCtime[idx+2])) + 
          (1. / (UTCtime[idx+1] - UTCtime[idx+3]));
    f0[1] = 1. - (2. * (tintp - UTCtime[idx+1]) * sum);
    sum = (1. / (UTCtime[idx+2] - UTCtime[idx])) + (1. / (UTCtime[idx+2] - UTCtime[idx+1])) + 
          (1. / (UTCtime[idx+2] - UTCtime[idx+3]));
    f0[2] = 1. - (2. * (tintp - UTCtime[idx+2]) * sum);
    sum = (1. / (UTCtime[idx+3] - UTCtime[idx])) + (1. / (UTCtime[idx+3] - UTCtime[idx+1])) + 
          (1. / (UTCtime[idx+3] - UTCtime[idx+2]));
    f0[3] = 1. - (2. * (tintp - UTCtime[idx+3]) * sum);

    double h[4] = {((tintp - UTCtime[idx+1]) / (UTCtime[idx] - UTCtime[idx+1])) * 
                   ((tintp - UTCtime[idx+2]) / (UTCtime[idx] - UTCtime[idx+2])) * 
                   ((tintp - UTCtime[idx+3]) / (UTCtime[idx] - UTCtime[idx+3])),
                   ((tintp - UTCtime[idx]) / (UTCtime[idx+1] - UTCtime[idx])) * 
                   ((tintp - UTCtime[idx+2]) / (UTCtime[idx+1] - UTCtime[idx+2])) * 
                   ((tintp - UTCtime[idx+3]) / (UTCtime[idx+1] - UTCtime[idx+3])),
                   ((tintp - UTCtime[idx]) / (UTCtime[idx+2] - UTCtime[idx])) * 
                   ((tintp - UTCtime[idx+1]) / (UTCtime[idx+2] - UTCtime[idx+1])) *
                   ((tintp - UTCtime[idx+3]) / (UTCtime[idx+2] - UTCtime[idx+3])),
                   ((tintp - UTCtime[idx]) / (UTCtime[idx+3] - UTCtime[idx])) * 
                   ((tintp - UTCtime[idx+1]) / (UTCtime[idx+3] - UTCtime[idx+1])) * 
                   ((tintp - UTCtime[idx+2]) / (UTCtime[idx+3] - UTCtime[idx+2]))};

    double hdot[4];
    sum = ((tintp - UTCtime[idx+2]) / (UTCtime[idx] - UTCtime[idx+2])) * 
          ((tintp - UTCtime[idx+3]) / (UTCtime[idx] - UTCtime[idx+3])) * 
          (1. / (UTCtime[idx] - UTCtime[idx+1]));
    sum += ((tintp - UTCtime[idx+1]) / (UTCtime[idx] - UTCtime[idx+1])) * 
           ((tintp - UTCtime[idx+3]) / (UTCtime[idx] - UTCtime[idx+3])) * 
           (1. / (UTCtime[idx] - UTCtime[idx+2]));
    sum += ((tintp - UTCtime[idx+1]) / (UTCtime[idx] - UTCtime[idx+1])) * 
           ((tintp - UTCtime[idx+2]) / (UTCtime[idx] - UTCtime[idx+2])) * 
           (1. / (UTCtime[idx] - UTCtime[idx+3]));
    hdot[0] = sum;

    sum = ((tintp - UTCtime[idx+2]) / (UTCtime[idx+1] - UTCtime[idx+2])) * 
          ((tintp - UTCtime[idx+3]) / (UTCtime[idx+1] - UTCtime[idx+3])) *
          (1. / (UTCtime[idx+1] - UTCtime[idx]));
    sum += ((tintp - UTCtime[idx]) / (UTCtime[idx+1] - UTCtime[idx])) * 
           ((tintp - UTCtime[idx+3]) / (UTCtime[idx+1] - UTCtime[idx+3])) * 
           (1. / (UTCtime[idx+1] - UTCtime[idx+2]));
    sum += ((tintp - UTCtime[idx]) / (UTCtime[idx+1] - UTCtime[idx])) * 
           ((tintp - UTCtime[idx+2]) / (UTCtime[idx+1] - UTCtime[idx+2])) *
           (1. / (UTCtime[idx+1] - UTCtime[idx+3]));
    hdot[1] = sum;

    sum = ((tintp - UTCtime[idx+1]) / (UTCtime[idx+2] - UTCtime[idx+1])) * 
          ((tintp - UTCtime[idx+3]) / (UTCtime[idx+2] - UTCtime[idx+3])) * 
          (1. / (UTCtime[idx+2] - UTCtime[idx]));
    sum += ((tintp - UTCtime[idx]) / (UTCtime[idx+2] - UTCtime[idx])) * 
           ((tintp - UTCtime[idx+3]) / (UTCtime[idx+2] - UTCtime[idx+3])) * 
           (1. / (UTCtime[idx+2] - UTCtime[idx+1]));
    sum += ((tintp - UTCtime[idx]) / (UTCtime[idx+2] - UTCtime[idx])) * 
           ((tintp - UTCtime[idx+1]) / (UTCtime[idx+2] - UTCtime[idx+1])) * 
           (1. / (UTCtime[idx+2] - UTCtime[idx+3]));
    hdot[2] = sum;

    sum = ((tintp - UTCtime[idx+1]) / (UTCtime[idx+3] - UTCtime[idx+1])) * 
          ((tintp - UTCtime[idx+2]) / (UTCtime[idx+3] - UTCtime[idx+2])) * 
          (1. / (UTCtime[idx+3] - UTCtime[idx]));
    sum += ((tintp - UTCtime[idx]) / (UTCtime[idx+3] - UTCtime[idx])) * 
           ((tintp - UTCtime[idx+2]) / (UTCtime[idx+3] - UTCtime[idx+2])) * 
           (1. / (UTCtime[idx+3] - UTCtime[idx+1]));
    sum += ((tintp - UTCtime[idx]) / (UTCtime[idx+3] - UTCtime[idx])) * 
           ((tintp - UTCtime[idx+1]) / (UTCtime[idx+3] - UTCtime[idx+1])) * 
           (1. / (UTCtime[idx+3] - UTCtime[idx+2]));
    hdot[3] = sum;

    double g0[4];
    double g1[4] = {h[0] + (2. * (tintp - UTCtime[idx]) * hdot[0]), 
                    h[1] + (2. * (tintp - UTCtime[idx+1]) * hdot[1]), 
                    h[2] + (2. * (tintp - UTCtime[idx+2]) * hdot[2]), 
                    h[3] + (2. * (tintp - UTCtime[idx+3]) * hdot[3])};
    sum = (1. / (UTCtime[idx] - UTCtime[idx+1])) + (1. / (UTCtime[idx] - UTCtime[idx+2])) + 
          (1. / (UTCtime[idx] - UTCtime[idx+3]));
    g0[0] = 2. * ((f0[0] * hdot[0]) - (h[0] * sum));
    sum = (1. / (UTCtime[idx+1] - UTCtime[idx])) + (1. / (UTCtime[idx+1] - UTCtime[idx+2])) + 
          (1. / (UTCtime[idx+1] - UTCtime[idx+3]));
    g0[1] = 2. * ((f0[1] * hdot[1]) - (h[1] * sum));
    sum = (1. / (UTCtime[idx+2] - UTCtime[idx])) + (1. / (UTCtime[idx+2] - UTCtime[idx+1])) + 
          (1. / (UTCtime[idx+2] - UTCtime[idx+3]));
    g0[2] = 2. * ((f0[2] * hdot[2]) - (h[2] * sum));
    sum = (1. / (UTCtime[idx+3] - UTCtime[idx])) + (1. / (UTCtime[idx+3] - UTCtime[idx+1])) +
          (1. / (UTCtime[idx+3] - UTCtime[idx+2]));
    g0[3] = 2. * ((f0[3] * hdot[3]) - (h[3] * sum));

    opos[0] = (((position[3*idx] * f0[0]) + (velocity[3*idx] * f1[0])) * h[0] * h[0]) + 
              (((position[3*(idx+1)] * f0[1]) + (velocity[3*(idx+1)] * f1[1])) * h[1] * h[1]) +
              (((position[3*(idx+2)] * f0[2]) + (velocity[3*(idx+2)] * f1[2])) * h[2] * h[2]) + 
              (((position[3*(idx+3)] * f0[3]) + (velocity[3*(idx+3)] * f1[3])) * h[3] * h[3]);
    opos[1] = (((position[3*idx+1] * f0[0]) + (velocity[3*idx+1] * f1[0])) * h[0] * h[0]) +
              (((position[3*(idx+1)+1] * f0[1]) + (velocity[3*(idx+1)+1] * f1[1])) * h[1] * h[1]) +
              (((position[3*(idx+2)+1] * f0[2]) + (velocity[3*(idx+2)+1] * f1[2])) * h[2] * h[2]) +
              (((position[3*(idx+3)+1] * f0[3]) + (velocity[3*(idx+3)+1] * f1[3])) * h[3] * h[3]);
    opos[2] = (((position[3*idx+2] * f0[0]) + (velocity[3*idx+2] * f1[0])) * h[0] * h[0]) +
              (((position[3*(idx+1)+2] * f0[1]) + (velocity[3*(idx+1)+2] * f1[1])) * h[1] * h[1]) +
              (((position[3*(idx+2)+2] * f0[2]) + (velocity[3*(idx+2)+2] * f1[2])) * h[2] * h[2]) +
              (((position[3*(idx+3)+2] * f0[3]) + (velocity[3*(idx+3)+2] * f1[3])) * h[3] * h[3]);

    ovel[0] = (((position[3*idx] * g0[0]) + (velocity[3*idx] * g1[0])) * h[0]) +
              (((position[3*(idx+1)] * g0[1]) + (velocity[3*(idx+1)] * g1[1])) * h[1]) +
              (((position[3*(idx+2)] * g0[2]) + (velocity[3*(idx+2)] * g1[2])) * h[2]) +
              (((position[3*(idx+3)] * g0[3]) + (velocity[3*(idx+3)] * g1[3])) * h[3]);
    ovel[1] = (((position[3*idx+1] * g0[0]) + (velocity[3*idx+1] * g1[0])) * h[0]) +
              (((position[3*(idx+1)+1] * g0[1]) + (velocity[3*(idx+1)+1] * g1[1])) * h[1]) +
              (((position[3*(idx+2)+1] * g0[2]) + (velocity[3*(idx+2)+1] * g1[2])) * h[2]) +
              (((position[3*(idx+3)+1] * g0[3]) + (velocity[3*(idx+3)+1] * g1[3])) * h[3]);
    ovel[2] = (((position[3*idx+2] * g0[0]) + (velocity[3*idx+2] * g1[0])) * h[0]) +
              (((position[3*(idx+1)+2] * g0[1]) + (velocity[3*(idx+1)+2] * g1[1])) * h[1]) +
              (((position[3*(idx+2)+2] * g0[2]) + (velocity[3*(idx+2)+2] * g1[2])) * h[2]) +
              (((position[3*(idx+3)+2] * g0[3]) + (velocity[3*(idx+3)+2] * g1[3])) * h[3]);

    return 0;
}

__device__ int gpuOrbit::interpolateLegendreOrbit(double tintp, double *opos, double *ovel) const {
    if (nVectors < 9) return 1;
    if ((tintp < UTCtime[0]) || (tintp > UTCtime[nVectors-1])) return 1;
    int idx = -1;
    for (int i=0; i<nVectors; i++) {
        if ((UTCtime[i] >= tintp) && (idx == -1)) {
            idx = min(max(i-5, 0), nVectors-9);
        }
    }

    double trel = (8. * (tintp - UTCtime[idx])) / (UTCtime[idx+8] - UTCtime[idx]);
    double teller = 1.;
    for (int i=0; i<9; i++) teller *= trel - i;
    if (teller == 0.) {
        opos[0] = position[3*(idx+int(trel))];
        opos[1] = position[3*(idx+int(trel))+1];
        opos[2] = position[3*(idx+int(trel))+2];
        ovel[0] = velocity[3*(idx+int(trel))];
        ovel[1] = velocity[3*(idx+int(trel))+1];
        ovel[2] = velocity[3*(idx+int(trel))+2];
    } else {
        double noemer[9] = {40320.0, -5040.0, 1440.0, -720.0, 576.0, -720.0, 1440.0, -5040.0, 
                            40320.0};
        opos[0] = opos[1] = opos[2] = 0.;
        ovel[0] = ovel[1] = ovel[2] = 0.;
        double coeff;
        for (int i=0; i<9; i++) {
            coeff = (teller / noemer[i]) / (trel - i);
            opos[0] += coeff * position[3*(idx+i)];
            opos[1] += coeff * position[3*(idx+i)+1];
            opos[2] += coeff * position[3*(idx+i)+2];
            ovel[0] += coeff * velocity[3*(idx+i)];
            ovel[1] += coeff * velocity[3*(idx+i)+1];
            ovel[2] += coeff * velocity[3*(idx+i)+2];
        }
    }
    return 0;
}

__device__ int gpuOrbit::interpolateSCHOrbit(double tintp, double *opos, double *ovel) const {
    if (nVectors < 2) return 1;
    if ((tintp < UTCtime[0]) || (tintp > UTCtime[nVectors-1])) return 1;
    opos[0] = opos[1] = opos[2] = 0.;
    ovel[0] = ovel[1] = ovel[2] = 0.;

    double frac;
    for (int i=0; i<nVectors; i++) {
        frac = 1.;
        for (int j=0; j<nVectors; j++) {
            if (i != j) frac *= (UTCtime[j] - tintp) / (UTCtime[j] - UTCtime[i]);
        }
        opos[0] += frac * position[3*i];
        opos[1] += frac * position[(3*i)+1];
        opos[2] += frac * position[(3*i)+2];
        ovel[0] += frac * velocity[3*i];
        ovel[1] += frac * velocity[(3*i)+1];
        ovel[2] += frac * velocity[(3*i)+2];

    }
    return 0;
}

__device__ int gpuOrbit::computeAcceleration(double tintp, double *acc) const {
    double dummy[3], vbef[3], vaft[3];
    if (interpolateWGS84Orbit(tintp-.01, dummy, vbef) == 1) return 1;
    if (interpolateWGS84Orbit(tintp+.01, dummy, vaft) == 1) return 1;
    acc[0] = (vaft[0] - vbef[0]) / .02;
    acc[1] = (vaft[1] - vbef[1]) / .02;
    acc[2] = (vaft[2] - vbef[2]) / .02;
    return 0;
}

__global__ void interpolateWGS84Orbit_d(gpuOrbit orb, double tintp, double *opos, double *ovel, 
                                        int *retcode) {
    /*
     *  GPU kernel to test interpolateWGS84Orbit() on the device for consistency. Since kernels must
     *  be void-type on the return, we store the return code in a variable that gets copied out.
     */
    *retcode = orb.interpolateWGS84Orbit(tintp, opos, ovel);
}

__host__ int gpuOrbit::interpolateWGS84Orbit_h(double tintp, cartesian_t &opos, 
                                               cartesian_t &ovel) {
    /*
     *  CPU-side function to call the corresponding GPU function on a single thread for consistency
     *  checking.
     */
    // Check inputs for valid length
    //checkVecLen(opos, 3);
    //checkVecLen(ovel, 3);
    // Malloc memory on the GPU and copy inputs over
    double *opos_d, *ovel_d;
    int *retcode_d;
    int retcode_h;
    cudaSetDevice(0);
    cudaMalloc((double**)&opos_d, 3*sizeof(double));
    cudaMalloc((double**)&ovel_d, 3*sizeof(double));
    cudaMalloc((int**)&retcode_d, sizeof(int));
    // Run the interpolateWGS84Orbit function on the gpuOrbit object on the GPU
    dim3 grid(1), block(1);
    interpolateWGS84Orbit_d <<<grid,block>>>(*this, tintp, opos_d, ovel_d, retcode_d);
    // Copy the results back to the CPU side and return any error code
    cudaMemcpy(opos.data(), opos_d, 3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ovel.data(), ovel_d, 3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&retcode_h, retcode_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(opos_d);
    cudaFree(ovel_d);
    return retcode_h;
}

__global__ void interpolateLegendreOrbit_d(gpuOrbit orb, double tintp, double *opos, double *ovel,
                                           int *retcode) {
    /*
     *  GPU kernel to test interpolateLegendreOrbit() on the device for consistency. Since kernels
     *  must be void-type on the return, we store the return code in a variable that gets copied
     *  out.
     */
    *retcode = orb.interpolateLegendreOrbit(tintp, opos, ovel);
}

__host__ int gpuOrbit::interpolateLegendreOrbit_h(double tintp, cartesian_t &opos,
                                               cartesian_t &ovel) {
    /*
     *  CPU-side function to call the corresponding GPU function on a single thread for consistency
     *  checking.
     */
    // Check inputs for valid length
    //checkVecLen(opos, 3);
    //checkVecLen(ovel, 3);
    // Malloc memory on the GPU and copy inputs over
    double *opos_d, *ovel_d;
    int *retcode_d;
    int retcode_h;
    cudaSetDevice(0);
    cudaMalloc((double**)&opos_d, 3*sizeof(double));
    cudaMalloc((double**)&ovel_d, 3*sizeof(double));
    cudaMalloc((int**)&retcode_d, sizeof(int));
    // Run the interpolateWGS84Orbit function on the gpuOrbit object on the GPU
    dim3 grid(1), block(1);
    interpolateWGS84Orbit_d <<<grid,block>>>(*this, tintp, opos_d, ovel_d, retcode_d);
    // Copy the results back to the CPU side and return any error code
    cudaMemcpy(opos.data(), opos_d, 3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ovel.data(), ovel_d, 3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&retcode_h, retcode_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(opos_d);
    cudaFree(ovel_d);
    return retcode_h;
}

__global__ void interpolateSCHOrbit_d(gpuOrbit orb, double tintp, double *opos, double *ovel,
                                      int *retcode) {
    /*
     *  GPU kernel to test interpolateSCHOrbit() on the device for consistency. Since kernels must
     *  be void-type on the return, we store the return code in a variable that gets copied out.
     */
    *retcode = orb.interpolateSCHOrbit(tintp, opos, ovel);
}

__host__ int gpuOrbit::interpolateSCHOrbit_h(double tintp, cartesian_t &opos,
                                               cartesian_t &ovel) {
    /*
     *  CPU-side function to call the corresponding GPU function on a single thread for consistency
     *  checking.
     */
    // Check inputs for valid length
    // checkVecLen(opos, 3);
    // checkVecLen(ovel, 3);
    // Malloc memory on the GPU and copy inputs over
    double *opos_d, *ovel_d;
    int *retcode_d;
    int retcode_h;
    cudaSetDevice(0);
    cudaMalloc((double**)&opos_d, 3*sizeof(double));
    cudaMalloc((double**)&ovel_d, 3*sizeof(double));
    cudaMalloc((int**)&retcode_d, sizeof(int));
    // Run the interpolateWGS84Orbit function on the gpuOrbit object on the GPU
    dim3 grid(1), block(1);
    interpolateWGS84Orbit_d <<<grid,block>>>(*this, tintp, opos_d, ovel_d, retcode_d);
    // Copy the results back to the CPU side and return any error code
    cudaMemcpy(opos.data(), opos_d, 3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ovel.data(), ovel_d, 3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&retcode_h, retcode_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(opos_d);
    cudaFree(ovel_d);
    return retcode_h;
}

