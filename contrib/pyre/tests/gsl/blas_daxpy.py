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
    a = 2
    x = 3
    y = 4
    # make a couple of vectors
    v1 = gsl.vector(shape=10).fill(x)
    v2 = gsl.vector(shape=10).fill(y)
    # compute the form
    gsl.blas.daxpy(a, v1, v2)
    # verify v1 was left alone
    assert v1 == gsl.vector(shape=10).fill(x)
    # and that v2 has the right value
    assert v2 == gsl.vector(shape=10).fill(a*x+y)
    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
