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
    azimuthNorm(poly.azimuthNorm)
{
    cudaSetDevice(0);
    
    const int n_coeffs = poly.coeffs.size();
    cudaMalloc((double**)&coeffs, n_coeffs*sizeof(double));
    cudaMemcpy(coeffs, &(poly.coeffs[0]), n_coeffs*sizeof(double), cudaMemcpyHostToDevice);
}
