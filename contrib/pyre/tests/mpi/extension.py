#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the extension module is accessible
"""


def test():
    # access the extension module
    import mpi
    # initialize it
    ext = mpi.init()
    # get the world communicator
    world = ext.world
    # extract the size of the communicator and my rank within it
    size = ext.communicatorSize(world)
    rank = ext.communicatorRank(world)
    # verify that my rank is within range
    assert rank in range(size)

    # for debugging purposes:
    # print("Hello from {}/{}!".format(rank, size))

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
