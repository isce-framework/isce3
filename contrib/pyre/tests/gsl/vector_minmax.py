#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Compute both the minimum and the maximum value of a vector
"""


def test():
    # package access
    import gsl
    # make a vector and initialize it
    v = gsl.vector(shape=100)
    # prime
    for index in range(v.shape): v[index] = 2*index+1
    # find both the min and the max
    small, big = v.minmax()
    # check it
    assert small == 1
    assert big == 2*(v.shape-1)+1
    # all done
    return v


# main
if __name__ == "__main__":
    test()


# end of file
