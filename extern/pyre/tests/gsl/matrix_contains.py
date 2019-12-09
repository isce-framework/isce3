#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that we can hunt down values in matrices
"""


def test():
    # package access
    import gsl
    # make a matrix
    m = gsl.matrix(shape=(100,100)).fill(1)
    # set an element to some value
    m[50,50] = 10
    # verify it happened
    assert 10 in m
    # all done
    return m


# main
if __name__ == "__main__":
    test()


# end of file
