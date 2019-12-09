#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that we can find specific values in a vector
"""


def test():
    # package access
    import gsl
    # make a vector
    v = gsl.vector(shape=100)
    # set an element to some value
    v[50] = 10
    # verify it happened
    assert v[50] == 10
    # check that it can be found
    assert 10 in v
    # all done
    return v


# main
if __name__ == "__main__":
    test()


# end of file
