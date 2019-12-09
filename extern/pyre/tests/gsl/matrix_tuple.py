#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify the tuple of tuples representation
"""


def test():
    # package access
    import gsl
    # make a matrix
    m = gsl.matrix(shape=(2,3))
    # set some values
    m[0,0] = 0
    m[0,1] = 1
    m[0,2] = 2
    m[1,0] = 10
    m[1,1] = 11
    m[1,2] = 12

    # verify the tuple rep
    assert m.tuple() == ( (0,1,2), (10,11,12) )

    # all done
    return m


# main
if __name__ == "__main__":
    test()


# end of file
