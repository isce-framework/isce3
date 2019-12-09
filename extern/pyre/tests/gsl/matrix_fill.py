#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Make a matrix and fill it with a constant
"""


def test():
    # package access
    import gsl

    # pick the matrix dimensions
    s1, s2 = 3, 4

    # make a matrix
    m1 = gsl.matrix(shape=(s1,s2))
    # set it to some value
    m1.fill(value=2)
    # verify it happened
    for e in m1: assert e == 2

    # make and initialize another matrix
    m2 = gsl.matrix(shape=(s1,s2)).zero()
    # fill it
    m2.fill(range(s1*s2))
    # verify
    assert m2.tuple() == tuple(tuple(range(n*s2, (n+1)*s2)) for n in range(s1))

    # all done
    return m1, m2


# main
if __name__ == "__main__":
    test()


# end of file
