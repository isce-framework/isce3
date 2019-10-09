#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise product reductions
"""


def test():
    # externals
    import mpi
    import math
    # initialize
    mpi.init()
    # get the world communicator
    world = mpi.world
    # and its structure
    rank = world.rank
    size = world.size
    # set up a destination for the reduction
    destination = int(size / 2)
    # create a value
    number = rank + 1
    # perform the reduction
    product = world.product(item=number, destination=destination)
    # check it
    if rank == destination:
        assert product == math.factorial(size)
    else:
        assert product is None
    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
