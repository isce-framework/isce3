#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise vector slicing
"""


def test():
    # package access
    import gsl
    # make a vector
    v = gsl.vector(shape=100)

    # fill it with our test pattern
    v[:] = range(v.shape)
    # verify it happened
    for i in range(v.shape): assert v[i] == i
    # and again
    assert tuple(v[:]) == tuple(range(v.shape))
    # also
    assert tuple(v[10:20:2]) == tuple(range(10,20,2))
    # and
    for value, i in zip(v[10::2], range(10,2)): assert value == i
    # all done
    return v


# main
if __name__ == "__main__":
    test()


# end of file
