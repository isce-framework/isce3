#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Lijun Zhu
# Caltech
# (c) 1998-2019 all rights reserved
#


"""
Exercise {dsymm}
the matrix-matrix product and sum for a symmetric matrix

C = \alpha A B + \beta C for Side is CblasLeft
C = \alpha B A + \beta C for Side is CblasRight
where A is symmetric.
When Uplo is CblasUpper, the upper triangle and diagonal of A are used;
when Uplo is CblasLower, the lower triangle and diagonal of A are used.
"""


def test():
    # get the package
    import gsl

    # the terms
    α = 2
    β = 3
    # the matrix A
    A = gsl.matrix(shape=(2,2))
    A[0,0], A[0,1] = 2,3
    A[1,0], A[1,1] = 3,2
    # the matrix B
    B = gsl.matrix(shape=(2,3))
    B[0,0], B[0,1], B[0,2] = 2,3,2
    B[1,0], B[1,1], B[1,2] = 1,2,1
    # the matrix C
    C = gsl.matrix(shape=(2,3))
    C[0,0], C[0,1], C[0,2] = 0,0,0
    C[1,0], C[1,1], C[1,2] = 0,0,1

    # compute the form
    gsl.blas.dsymm(A.sideLeft, A.upperTriangular, α, A, B, β, C)

    # check
    assert tuple(C) == (14, 24, 14, 16, 26, 19)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
