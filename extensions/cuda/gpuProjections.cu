//
// Source Author: Paulo Penteado, based on Projections.cpp by Piyush Agram / Joshua Cohen
// Copyright 2018
//

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include "Projections.h"
#include "gpuProjections.h"
using isce::core::cartesian_t;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
       std::cerr<< "GPUassert: " << cudaGetErrorString(code) << " " << file << "  " << line << "\n";
      if (abort) exit(code);
   }
}

namespace isce {
    namespace cuda {
        namespace core {

//Helper for the host side function
__global__ void forward_g(int code,
                          ProjectionBase** base,
                          const double *inpts,
                          double *outpts,
                          int *flags)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*base) = new PolarStereo(code);
        flags[0] = (*base)->forward(inpts, outpts);
        delete *base;
    }
}

//Helper for the host side function
__global__ void inverse_g(int code,
                          ProjectionBase **base,
                          const double *inpts,
                          double *outpts,
                          int *flags)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*base) = new PolarStereo(code);
        flags[0] = (*base)->inverse(inpts, outpts);
        delete *base;
    }
}

__host__ int ProjectionBase::forward_h(const cartesian_t &llh, cartesian_t &xyz) const
{
    double *llh_d, *xyz_d;
    int *flag_d;
    ProjectionBase **base_d;

    gpuErrChk( cudaSetDevice(7));
    gpuErrChk( cudaMalloc((int**)&flag_d, 1*sizeof(int)));
    gpuErrChk( cudaMalloc((double**)&llh_d,3*sizeof(double)));
    gpuErrChk( cudaMalloc((double**)&xyz_d,3*sizeof(double)));
    gpuErrChk( cudaMalloc(&base_d, sizeof(ProjectionBase**)));
    gpuErrChk( cudaMemcpy(llh_d, llh.data(), 3*sizeof(double), cudaMemcpyHostToDevice));

    //Call the global function with a single thread
    forward_g<<<1,1>>>(_epsgcode, base_d, llh_d, xyz_d, flag_d);
    gpuErrChk(cudaDeviceSynchronize());
    gpuErrChk( cudaMemcpy(xyz.data(), xyz_d, 3*sizeof(double), cudaMemcpyDeviceToHost));
    int status;
    gpuErrChk( cudaMemcpy(&status, flag_d, sizeof(int), cudaMemcpyDeviceToHost));
    //Clean up
    gpuErrChk( cudaFree(llh_d));
    gpuErrChk( cudaFree(xyz_d));
    gpuErrChk( cudaFree(flag_d));
    gpuErrChk( cudaFree(base_d));
    return status;
}

__host__ int ProjectionBase::inverse_h(const cartesian_t &xyz, cartesian_t &llh) const
{
    double *llh_d, *xyz_d;
    int *flag_d;
    ProjectionBase **base_d;
    gpuErrChk( cudaSetDevice(7));
    gpuErrChk( cudaMalloc((int**)&flag_d, sizeof(int)));
    gpuErrChk( cudaMalloc((double**)&llh_d,3*sizeof(double)));
    gpuErrChk( cudaMalloc((double**)&xyz_d,3*sizeof(double)));
    gpuErrChk( cudaMalloc(&base_d, sizeof(ProjectionBase**)));
    gpuErrChk( cudaMemcpy(xyz_d, xyz.data(), 3*sizeof(double), cudaMemcpyHostToDevice));

    //Call the global function with a single thread
    dim3 grid(1), block(1);
    inverse_g<<<grid,block>>>(_epsgcode*1, base_d, xyz_d, llh_d, flag_d);
    gpuErrChk( cudaDeviceSynchronize());
    gpuErrChk( cudaMemcpy(llh.data(), llh_d, 3*sizeof(double), cudaMemcpyDeviceToHost));
    int status;
    gpuErrChk( cudaMemcpy(&status, flag_d, sizeof(int), cudaMemcpyDeviceToHost));

    //Clean up
    gpuErrChk( cudaFree(llh_d));
    gpuErrChk( cudaFree(xyz_d));
    gpuErrChk( cudaFree(flag_d));
    gpuErrChk( cudaFree(base_d));
    return status;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* * * * * * * * * * * * * * * * * * * PolarStereo Projection * * * * * * * * * * * * * * * * * * */
__host__ __device__ double pj_tsfn(double phi, double sinphi, double e) {
    /*
     * Local function - Determine small t from PROJ.4.
     */
    sinphi *= e;
    return tan(.5 * ((.5*M_PI) - phi)) / pow((1. - sinphi) / (1. + sinphi), .5*e);
}


__host__ __device__ PolarStereo::PolarStereo(int code) : ProjectionBase(code) {
    /*
     * Set up various parameters for polar stereographic projection. Currently only EPSG:3031
     * (Antarctic) and EPSG:3413 (Greenland) are supported.
     */
    if (_epsgcode == 3031) {
        printf("Antarctica \n");
        isnorth = false;
        //lat0 = -M_PI / 2.;
        // Only need absolute value
        lat_ts = (71. * M_PI) / 180.;
        lon0 = 0.;
    } else if (_epsgcode == 3413) {
        printf("Greenland \n");
        isnorth = true;
        //lat0 = M_PI / 2.;
        lat_ts = 70. * (M_PI / 180.);
        lon0 = -45. * (M_PI / 180.);
    } else {
            //Need to figure out a way to throw error on device
            //Currently, delegated to CPU side
    }
    e = sqrt(ellipse.gete2());
    akm1 = cos(lat_ts) / pj_tsfn(lat_ts, sin(lat_ts), e);
    akm1 *= ellipse.geta() / sqrt(1. - (pow(e,2) * pow(sin(lat_ts),2)));
}

__device__ int PolarStereo::forward(const double *llh, double *out)  const{
    /*
     * Transform from LLH to Polar Stereo.
     * CUDA device function
     * TODO: organize shareable variables
     * TODO: test for numerical exceptions and out of bounds coordinates
     */
    printf("Inside polar \n");
    double lam = llh[0] - lon0;
    double phi = llh[1] * (isnorth ? 1. : -1.);
    double temp = akm1 * pj_tsfn(phi, sin(phi), e);

    out[0] = temp * sin(lam);
    out[1] = -temp * cos(lam) * (isnorth ? 1. : -1.);
    //Height is just pass through
    out[2] = llh[2];
    return 0;
	
}

__device__ int PolarStereo::inverse(const double *ups, double *llh) const {
    /*
     * Transform from Polar Stereo to LLH.
     * CUDA device function
     * TODO: organize shareable variables
     * TODO: find out how many iterations allow getting rid of the conditional,
     * replaced by a constant number of iterations.
     * TODO: test for numerical exceptions and out of bounds coordinates
     */
    double tp = -hypot(ups[0], ups[1])/akm1;
    double fact = (isnorth)?1:-1;
    double phi_l = (.5*M_PI) - (2. * atan(tp));

    double sinphi;
    double phi = 0.;
    for(int i=8; i--; phi_l = phi) {
        sinphi = e * sin(phi_l);
        phi = 2. * atan(tp * pow((1. + sinphi) / (1. - sinphi), -0.5*e)) +0.5 * M_PI;
        if (fabs(phi_l - phi) < 1.e-10) {
            llh[0] = ((ups[0] == 0.) && (ups[1] == 0.)) ? 0. : atan2(ups[0], -fact*ups[1]) + lon0;
            llh[1] = phi*fact;
            llh[2] = ups[2];
            return 0;
        }
    }
    return 1;

}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* * * * * * * * * * * * * * * * * * * Projection Transformer * * * * * * * * * * * * * * * * * * */
__host__ int projTransform_h(ProjectionBase &in, ProjectionBase &out, const cartesian_t &inpts,
                  cartesian_t &outpts) {
    if (in._epsgcode == out._epsgcode) {
        // If input/output projections are the same don't even bother processing
        outpts = inpts;
        return 0;
    } else if (in._epsgcode == 4326) {
        // Consider case where input is Lat/Lon
        return out.forward_h(inpts, outpts);
    } else if (out._epsgcode == 4326) {
        // Consider case where output is Lat/Lon
        return -out.inverse_h(inpts, outpts);
    } else {
        cartesian_t temp;
        if (in.inverse_h(inpts, temp) != 0) return -2;
        if (out.forward_h(temp, outpts) != 0) return 2;
    }
    return 0;
};
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
        }
    }
}
