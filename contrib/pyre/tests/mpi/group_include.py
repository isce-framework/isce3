#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the group manipulation interface
"""


def test():
    # access the package
    import mpi
    # initialize
    mpi.init()
    # grab the world communicator
    world = mpi.world
    # access the world group
    whole = world.group()
    # slice just the even ranks
    evens = whole.include(rank for rank in range(world.size) if (rank % 2 == 0))

    # check that the size of this group is half the number of processors
    assert evens.size == (world.size+1) // 2

    # and check my rank
    if world.rank % 2 == 0:
        assert evens.rank == world.rank / 2
    else:
        assert evens.rank == evens.mpi.undefined

    return


# main
if __name__ == "__main__":
    test()


# end of file
