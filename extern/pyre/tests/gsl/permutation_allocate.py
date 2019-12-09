#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Create a permutation
"""


def test():
    # get the package
    import gsl
    # make a permutation
    p = gsl.permutation(shape=100).init()
    # check that we got a valid permutation back
    assert p
    # and return it
    return p


# main
if __name__ == "__main__":
    test()


# end of file
