# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
"""
Support for the linear algebra interface
"""

# externals
from . import gsl
from .Matrix import Matrix
from .Permutation import Permutation


# LU
def LU_decomposition(matrix):
    """
    Compute the LU decomposition of a matrix
    """
    # get the triplet (matrix, permutation, sign) that results from the LU decomposition
    _, pcapsule, sign = gsl.linalg_LU_decomp(matrix.data)
    # build a wrapper for the permutation
    p = Permutation(shape=matrix.rows, data=pcapsule)
    # return the triplet
    return (matrix, p, sign)


def LU_invert(matrix, permutation, sign):
    """
    Compute the inverse of {matrix} given its LU decomposition; a new matrix is returned
    """
    # invert; the result is a matrix data capsule
    capsule = gsl.linalg_LU_invert(matrix.data, permutation.data)
    # build a matrix from it
    return Matrix(shape=matrix.shape, data=capsule)


def LU_det(matrix, permutation, sign):
    """
    Compute the determinant of {matrix} given its LU decomposition
    """
    # easy enough
    return gsl.linalg_LU_det(matrix.data, sign)


def LU_lndet(matrix, permutation, sign):
    """
    Compute the determinant of {matrix} given its LU decomposition
    """
    # easy enough
    return gsl.linalg_LU_lndet(matrix.data)


# Cholesky
def cholesky_decomposition(matrix):
    """
    Compute the Cholesky decomposition of a symmetric positive definite matrix
    """
    # compute the decomposition
    gsl.linalg_cholesky_decomp(matrix.data)
    # and return the matrix
    return matrix


# end of file
