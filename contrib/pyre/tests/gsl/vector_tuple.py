#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify the tuple representation
"""


def test():
    # package access
    import gsl
    # make a matrix
    v = gsl.vector(shape=3)
    # set some values
    v[0] = 0
    v[1] = 1
    v[2] = 2

    # verify the tuple rep
    assert v.tuple() == (0,1,2)

    # all done
    return v


# main
if __name__ == "__main__":
    test()


# end of file
