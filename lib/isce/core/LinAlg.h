//
// Author: Joshua Cohen, Bryan V. Riel
// Copyright 2017
//

#ifndef ISCE_CORE_LINALG_H
#define ISCE_CORE_LINALG_H

#include <vector>
#include "Constants.h"

// Declaration
namespace isce {
    namespace core {
        struct LinAlg;
    }
}

// LinAlg declaration
struct isce::core::LinAlg {
    LinAlg() = default;
    // Cross product
    static void cross(const cartesian_t &, const cartesian_t &, cartesian_t &);
    // Dot product
    static double dot(const cartesian_t &,const cartesian_t &);
    // Linear combination of vectors
    static void linComb(double, const cartesian_t &, double, const cartesian_t &,
                        cartesian_t &);
    // Matrix-matrix multiplication
    static void matMat(const cartmat_t &, const cartmat_t &, cartmat_t &);
    // Matrix-vector multiplication
    static void matVec(const cartmat_t &, const cartesian_t &, cartesian_t &);
    // Norm of vector
    static double norm(const cartesian_t &);
    // Transpose matrix
    static void tranMat(const cartmat_t &, cartmat_t &);
    // Unit vector
    static void unitVec(const cartesian_t &, cartesian_t &);
    // Compute ENU basis
    static void enuBasis(double, double, cartmat_t &);
};

#endif

// end of file
