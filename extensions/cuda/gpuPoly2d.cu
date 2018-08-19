//
// Author: Liang Yu
// Copyright 2018
//
// NOTE: gpuOrbit used as template

#include <cuda_runtime.h>
#include <vector>
#include "Constants.h"
#include "gpuPoly2d.h"
#include "Poly2d.h"

using isce::core::cuda::gpuPoly2d;
using isce::core::Poly2d;
using std::vector;

__host__ gpuPoly2d::gpuPoly2d(const Poly2d &poly) :
    rangeOrder(poly.rangeOrder), 
    azimuthOrder(poly.azimuthOrder), 
    rangeMean(poly.rangeMean), 
    azimuthMean(poly.azimuthMean),
    rangeNorm(poly.rangeNorm), 
    azimuthNorm(poly.azimuthNorm),
    owner(true)
{
    cudaSetDevice(0);
    
    const int n_coeffs = poly.coeffs.size();
    cudaMalloc((double**)&coeffs, n_coeffs*sizeof(double));
    cudaMemcpy(coeffs, &(poly.coeffs[0]), n_coeffs*sizeof(double), cudaMemcpyHostToDevice);
}


gpuPoly2d::~gpuPoly2d() {
    if (owner) {
        cudaFree(coeffs);
    }
}

__device__ double gpuPoly2d::eval(double azi, double rng) {

    double xval = (rng - rangeMean) / rangeNorm;
    double yval = (azi - azimuthMean) / azimuthNorm;

    double scalex;
    double scaley = 1.;
    double val = 0.;
    for (int i=0; i<=azimuthOrder; i++,scaley*=yval) {
        scalex = 1.;
        for (int j=0; j<=rangeOrder; j++,scalex*=xval) {
            val += scalex * scaley * coeffs[IDX1D(i,j,rangeOrder+1)];
        }
    }
    return val;
}

__global__ void eval_d(gpuPoly2d p, double azi, double rng, double *val)
{
    *val = p.eval(azi, rng);
}

__host__ double gpuPoly2d::eval_h(double azi, double rng)
{
    cudaSetDevice(0);
    double *val_d;
    double val_h;
    // use unified memory?
    cudaMalloc((double**)&val_d, sizeof(double));
    dim3 grid(1), block(1);
    eval_d<<<grid,block>>>(*this, azi, rng, val_d);
    cudaMemcpy(&val_h, val_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(val_d);
    return val_h;
}

/*
void isce::core::Poly2d::
setCoeff(int row, int col, double val) {
    if ((row < 0) || (row > azimuthOrder)) {
        std::string errstr = "Poly2d::setCoeff - Trying to set coefficient for row " + 
                             std::to_string(row+1) + " out of " + 
                             std::to_string(azimuthOrder+1);
        throw std::out_of_range(errstr);
    }
    if ((col < 0) || (col > rangeOrder)) {
        std::string errstr = "Poly2d::setCoeff - Trying to set coefficient for col " +
                             std::to_string(col+1) + " out of " + std::to_string(rangeOrder+1);
        throw std::out_of_range(errstr);
    }
    coeffs[IDX1D(row,col,rangeOrder+1)] = val;
}

double isce::core::Poly2d::
getCoeff(int row, int col) const {
    if ((row < 0) || (row > azimuthOrder)) {
        std::string errstr = "Poly2d::getCoeff - Trying to get coefficient for row " +
                             std::to_string(row+1) + " out of " + 
                             std::to_string(azimuthOrder+1);
        throw std::out_of_range(errstr);
    }
    if ((col < 0) || (col > rangeOrder)) {
        std::string errstr = "Poly2d::getCoeff - Trying to get coefficient for col " + 
                             std::to_string(col+1) + " out of " + std::to_string(rangeOrder+1);
        throw std::out_of_range(errstr);
    }
    return coeffs[IDX1D(row,col,rangeOrder+1)];
}
*/
