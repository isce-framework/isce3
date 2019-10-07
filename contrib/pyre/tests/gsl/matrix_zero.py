#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Make a matrix and fill it with zeroes
"""


def test():
    # package access
    import gsl
    # make a matrix
    m = gsl.matrix(shape=(100,100))
    # zero it out
    m.zero()
    # verify it happened
    for e in m: assert e == 0
    # all done
    return m


# main
if __name__ == "__main__":
    test()


# end of file
