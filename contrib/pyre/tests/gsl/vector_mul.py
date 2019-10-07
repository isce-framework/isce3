#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise in-place, element by element vector multiplication
"""


def test():
    # package access
    import gsl
    # make a couple of vectors and initialize them
    v1 = gsl.vector(shape=100).fill(value=2)
    v2 = gsl.vector(shape=100).fill(value=2)
    # check
    for e in v1: assert e == 2
    for e in v2: assert e == 2
    # multiply them and store the result in v1
    v1 *= v2
    # check
    for e in v1: assert e == 4
    for e in v2: assert e == 2
    # all done
    return v1, v2


# main
if __name__ == "__main__":
    test()


# end of file
