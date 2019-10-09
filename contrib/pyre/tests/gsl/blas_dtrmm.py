#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise {dtrmm}
"""

# get the package
import gsl


def test():
    # make a lower triangular matrix
    A = lower()
    # show me
    # A.print()

    # make a sample matrix
    B = full()
    # show me
    # B.print()

    # try one of the supported operations
    B = gsl.blas.dtrmm(
        A.sideLeft, A.lowerTriangular, A.opNoTrans, A.unitDiagonal,
        2, A, B
        )
    # show me
    # B.print()
    # the expected result
    result = gsl.matrix(shape=(3,3))
    result[0,0] = 0
    result[0,1] = 2
    result[0,2] = 4
    result[1,0] = 6
    result[1,1] = 12
    result[1,2] = 18
    result[2,0] = 36
    result[2,1] = 52
    result[2,2] = 68

    # check
    assert B == result

    # make an upper triangular matrix
    A = upper()
    # show me
    # A.print()

    # make a sample matrix
    B = full()
    # show me
    # B.print()

    # try one of the supported operations
    B = gsl.blas.dtrmm(
        A.sideLeft, A.upperTriangular, A.opNoTrans, A.unitDiagonal,
        2, A, B
        )
    # show me
    # B.print()
    # the expected result
    result = gsl.matrix(shape=(3,3))
    result[0,0] = 48
    result[0,1] = 60
    result[0,2] = 72
    result[1,0] = 54
    result[1,1] = 64
    result[1,2] = 74
    result[2,0] = 12
    result[2,1] = 14
    result[2,2] = 16

    # check
    assert B == result
    # all done
    return


def full():
    """
    Build a sample matrix
    """
    # make one
    m = gsl.matrix(shape=(3,3))
    # fill it
    for i in range(3):
        for j in range(3):
            m[i,j] = 3*i + j
    # all done
    return m


def lower():
    # make a triangular matrix
    m = gsl.matrix(shape=(3,3)).identity()
    # that's non-trivial
    m[1,0] = 2
    m[2,0] = 3
    m[2,1] = 4
    # all done
    return m


def upper():
    # make a triangular matrix
    m = gsl.matrix(shape=(3,3)).identity()
    # that's non-trivial
    m[0,1] = 2
    m[0,2] = 3
    m[1,2] = 4
    # all done
    return m


# main
if __name__ == "__main__":
    test()


# end of file
