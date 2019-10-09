#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise finding the maximum value of a matrix
"""


def test():
    # package access
    import gsl
    # make a matrix and initialize it
    m = gsl.matrix(shape=(100,100)).fill(value=1)
    # unpack the shape
    s0, s1 = m.shape
    # prime
    for i0 in range(s0):
        for i1 in range(s1):
            m[i0, i1] = 2*(i0+i1)+1
    # find the min
    small = m.min()
    # check it
    assert small == 1
    # all done
    return m


# main
if __name__ == "__main__":
    test()


# end of file
