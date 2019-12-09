#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the linalg package is accessible
"""


def test():
    # get the package
    import gsl

    # create a trivial matrix
    one = gsl.matrix(shape=(100,100)).identity()

    # clone it
    one_c = one.clone()
    # decompose it
    decomp = gsl.linalg.LU_decomposition(one_c)
    # check
    assert type(decomp[0]) == gsl.matrix
    assert type(decomp[1]) == gsl.permutation
    assert type(decomp[2]) == int

    # invert it
    inv = gsl.linalg.LU_invert(*decomp)
    # check it
    for i in range(inv.rows):
        for j in range(inv.columns):
            if i == j:
                assert one[i,j] == 1
            else:
                assert one[i,j] == 0

    # compute the determinant
    det = gsl.linalg.LU_det(*decomp)
    # check it
    assert det == 1

    # return the decomposition
    return decomp


# main
if __name__ == "__main__":
    test()


# end of file
