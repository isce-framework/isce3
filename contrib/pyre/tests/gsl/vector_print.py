#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Make a vector, fill it with a constant, and print it
"""


def test():
    # package access
    import gsl
    # make a matrix
    v = gsl.vector(shape=3)
    # set it to some value
    v.fill(value=2)
    # print it
    v.print(indent=' '*4)
    # all done
    return v


# main
if __name__ == "__main__":
    test()


# end of file
