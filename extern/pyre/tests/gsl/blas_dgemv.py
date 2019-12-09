#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise {dgemv}
"""


def test():
    # get the package
    import gsl

    # the terms
    α = 2
    β = 3
    # the vector x
    x = gsl.vector(shape=3)
    x[0], x[1], x[2] = 1,2,3
    # the vector y
    y = gsl.vector(shape=3)
    y[0], y[1], y[2] = 2,4,6
    # the matrix A
    A = gsl.matrix(shape=(3,3)).identity()
    A[0,1], A[0,2], A[1,2] = 2,3,2
    A[1,0], A[2,0], A[2,1] = 2,3,2

    # compute the form
    y = gsl.blas.dgemv(A.opNoTrans, α, A, x, β, y)

    # check
    # print(tuple(y))
    assert tuple(y) == (34, 32, 38)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
