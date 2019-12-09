#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Fill a vector with zeroes
"""


def test():
    # package access
    import gsl
    # make a vector
    v = gsl.vector(shape=100)
    # zero it out
    v.zero()
    # verify it happened
    for e in v: assert e == 0
    # all done
    return v


# main
if __name__ == "__main__":
    test()


# end of file
