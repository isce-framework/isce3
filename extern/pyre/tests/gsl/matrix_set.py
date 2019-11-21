#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Set a matrix element to some value and verify it can be retrieved correctly
"""


def test():
    # package access
    import gsl
    # make a matrix
    m = gsl.matrix(shape=(100, 100))
    # fill it with zeroes
    m.fill(0)
    # set an element to some value
    m[50,50] = 10
    # verify it happened
    assert m[50,50] == 10

    # set another
    m[99,99] = 5
    # access using reflected indices
    assert m[-1, -1] == 5

    # out of bounds get
    try:
        m[500,500]
        assert False
    except IndexError:
        pass

    # out of bounds set
    try:
        m[500,500] = 1
        assert False
    except IndexError:
        pass

    # reflected out of bounds get
    try:
        m[-500,-500]
        assert False
    except IndexError:
        pass

    # reflected out of bounds set
    try:
        m[-500,-500] = 1
        assert False
    except IndexError:
        pass

    # bad index tuples
    try:
        m[1,2,3]
        assert False
    except TypeError:
        pass
    # and
    try:
        m[1,2,3] = 0
        assert False
    except TypeError:
        pass

    # all done
    return m


# main
if __name__ == "__main__":
    test()


# end of file
