#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Allocate a matrix of a given shape
"""


def test():
    # package access
    import gsl
    # make a matrix
    m = gsl.matrix(shape=(100,100))
    # all done
    return m


# main
if __name__ == "__main__":
    test()


# end of file
