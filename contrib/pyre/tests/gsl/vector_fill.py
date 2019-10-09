#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Fill a vector with a constant value
"""


def test():
    # package access
    import gsl

    # make a vector
    v1 = gsl.vector(shape=100)
    # set it to some value
    v1.fill(value=2)
    # verify it happened
    for e in v1: assert e == 2

    # make and initialize another vector
    v2 = gsl.vector(shape=100).zero()
    # fill it
    v2.fill(range(100))
    # verify
    assert v2.tuple() == tuple(range(100))

    # all done
    return v1, v2


# main
if __name__ == "__main__":
    test()


# end of file
