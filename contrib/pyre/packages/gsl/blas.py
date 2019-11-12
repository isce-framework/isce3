# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
"""
Support for the BLAS interface
"""

# externals
from . import gsl


# the interface for doubles
# level 1
def ddot(x, y):
    """
    Compute the scalar product {x^T y}
    """
    # compute and return the result
    return gsl.blas_ddot(x.data, y.data)


def dnrm2(x):
    """
    Compute the Euclidean norm
    """
    # compute and return the result
    return gsl.blas_dnrm2(x.data)


def dasum(x):
    """
    Compute the sum of the absolute values of the entries in {x}
    """
    # compute and return the result
    return gsl.blas_dasum(x.data)


def idamax(x):
    """
    Return the index with the largest value in {x}
    """
    # compute and return the result
    return gsl.blas_idamax(x.data)


def dswap(x, y):
    """
    Exchange the contents of {x} and {y}
    """
    # compute
    gsl.blas_dswap(x.data, y.data)
    # and return {x} and {y}
    return x, y


def dcopy(x, y):
    """
    Copy the elements of {x} into {y}
    """
    # compute
    gsl.blas_dcopy(x.data, y.data)
    # and return {x} and {y}
    return y


def daxpy(α, x, y):
    """
    Compute {α x + y} and store the result in {y}
    """
    # compute
    gsl.blas_daxpy(α, x.data, y.data)
    # and return the result {y}
    return y


def dscal(α, x):
    """
    Compute x = α x
    """
    # compute
    gsl.blas_dscal(α, x.data)
    # and return {x}
    return x


def drotg(x, y):
    """
    Compute the Givens rotation which zeroes the vectors {x} and {y}
    """
    # compute and return the tuple (x, y, c,s) for the Givens rotation
    return gsl.blas_drotg(x, y)


def drot(x, y, c, s):
    """
    Apply the Givens rotation {(c,s)} to {x} and {y}
    """
    # compute
    gsl.blas_drot(x.data, y.data, c, s)
    # and return
    return x, y


# level 2
def dgemv(transpose, α, A, x, β, y):
    """
    Compute {y = α op(A) x + β y}
    """
    # compute
    gsl.blas_dgemv(transpose, α, A.data, x.data, β, y.data)
    # and return the result
    return y


def dtrmv(uplo, transpose, diag, A, x):
    """
    Compute {x = op(A) x}
    """
    # compute
    gsl.blas_dtrmv(uplo, transpose, diag, A.data, x.data)
    # and return the result
    return x


def dtrsv(uplo, transpose, diag, A, x):
    """
    Compute {x = inv(op(A)) x}
    """
    # compute
    gsl.blas_dtrsv(uplo, transpose, diag, A.data, x.data)
    # and return the result
    return x


def dsymv(uplo, α, A, x, β, y):
    """
    Compute {y = α A x + β y}
    """
    # compute
    gsl.blas_dsymv(uplo, α, A.data, x.data, β, y.data)
    # and return the result in {y}
    return y


def dsyr(uplo, α, x, A):
    """
    Compute {A = α x x^T + A}
    """
    # compute
    gsl.blas_dsyr(uplo, α, x.data, A.data)
    # and return the result in {A}
    return A


# level 3
def dgemm(tranA, tranB, α, A, B, β, C):
    """
    Compute {C = α op(A) op(B) + β C}
    """
    # compute
    gsl.blas_dgemm(tranA, tranB, α, A.data, B.data, β, C.data)
    # and return the result
    return C

def dsymm(side, uploA, α, A, B, β, C):
    """
    Compute {C = α A B + β C} or {C = α B A + β C} depending on {side}, A is symmetric
    """
    #compute
    gsl.blas_dsymm(side, uploA, α, A.data, B.data, β, C.data)
    #return the result
    return C

def dtrmm(sideA, uplo, transpose, diag, α, A, B):
    """
    Compute {B = α op(A) B} or {B = α B op(A)} depending on the value of {sideA}
    """
    # compute
    gsl.blas_dtrmm(sideA, uplo, transpose, diag, α, A.data, B.data)
    # and return the result
    return B


# end of file
