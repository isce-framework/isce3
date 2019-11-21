#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise max reductions
"""


def test():
    # access the package
    import mpi
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
    number = rank**2
    # perform the reduction
    largest = world.max(item=number, destination=destination)
    # check it
    if rank == destination:
        assert largest == (size-1)**2
    else:
        assert largest is None
    # perform the all process reduction
    largest = world.max(item=number)
    # check it
    assert largest == (size-1)**2
    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
