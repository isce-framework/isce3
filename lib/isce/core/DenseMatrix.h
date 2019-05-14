#pragma once
#ifndef ISCE_CORE_DENSEMATRIX_H
#define ISCE_CORE_DENSEMATRIX_H

#include "Vector.h"

namespace isce { namespace core {
    template<int N>
    struct DenseMatrix;

    typedef DenseMatrix<3> Mat3;
}}

template<int N>
struct isce::core::DenseMatrix {

    Vector<N> data[N];

    CUDA_HOSTDEV DenseMatrix() {}

    CUDA_HOSTDEV DenseMatrix(std::initializer_list<std::initializer_list<double>> lst) {
        int i = 0, j = 0;
        for (const auto& l : lst) {
            for (const auto& v : l) {
                data[i][j++] = v;
            }
            i++, j = 0;
        }
    }

    CUDA_HOSTDEV           Vector<N>& operator[](int i)       { return data[i]; }
    CUDA_HOSTDEV constexpr Vector<N>  operator[](int i) const { return data[i]; }
};

#endif
