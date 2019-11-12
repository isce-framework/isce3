#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise {ddot}
"""


def test():
    # get the package
    import gsl
    # a couple of values
    x = 3
    y = 2
    # make a couple of vectors
    v1 = gsl.vector(shape=10).fill(x)
    v2 = gsl.vector(shape=10).fill(y)
    # compute the dot product
    assert gsl.blas.ddot(v1, v2) ==  x*y*v1.shape
    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
