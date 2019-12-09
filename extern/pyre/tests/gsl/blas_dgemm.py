#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise {dgemm}
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
    A[1,0], A[1,1] = 1,2
    # the matrix B
    B = gsl.matrix(shape=(2,3))
    B[0,0], B[0,1], B[0,2] = 2,3,2
    B[1,0], B[1,1], B[1,2] = 1,2,1
    # the matrix C
    C = gsl.matrix(shape=(2,3))
    C[0,0], C[0,1], C[0,2] = 0,0,0
    C[1,0], C[1,1], C[1,2] = 0,0,0

    # compute the form
    gsl.blas.dgemm(A.opNoTrans, B.opNoTrans, α, A, B, β, C)

    # check
    # print(tuple(y))
    assert tuple(C) == (14, 24, 14, 8, 14, 8)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
