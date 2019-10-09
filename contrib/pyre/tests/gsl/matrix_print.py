#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Make a matrix, fill it with a constant, and print it
"""


def test():
    # package access
    import gsl
    # make a matrix
    m = gsl.matrix(shape=(3,3))
    # set it to some value
    m.fill(value=2)
    # print it
    m.print(indent=' '*4)
    # all done
    return m


# main
if __name__ == "__main__":
    test()


# end of file
