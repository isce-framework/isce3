#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise setting and getting individual vector elements
"""


def test():
    # package access
    import gsl
    # make a vector
    v = gsl.vector(shape=100)
    # fill with a test pattern
    for i in range(len(v)): v[i] = i
    # verify it happened
    assert v[50] == 50

    # access through reflection
    v[-99] == v[1]

    # all done
    return v


# main
if __name__ == "__main__":
    test()


# end of file
