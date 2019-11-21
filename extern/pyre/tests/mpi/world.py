#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the world communicator is accessible
"""


def test():
    # access the package
    import mpi
    # initialize it
    mpi.init()
    # get the world communicator
    world = mpi.world
    # extract the size of the communicator and my rank within it
    size = world.size
    rank = world.rank
    # verify that my rank is within range
    assert rank in range(size)

    # for debugging purposes:
    # import platform
    # print("Hello from {}/{}: {}".format(rank, size, platform.node()))

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
