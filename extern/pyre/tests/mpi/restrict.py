#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise communicator construction given a process group
"""


def test():
    # access the package
    import mpi
    # initialize
    mpi.init()
    # grab the world communicator
    world = mpi.world
    # access the world process group
    whole = world.group()
    # build a tuple of the even ranks
    ranks = tuple(rank for rank in range(world.size) if (rank % 2 == 0))
    # convert it into a group
    even = whole.include(ranks)

    # build a group with the odd ranks from the difference (world - even)
    odd = whole.difference(even)
    # and the matching communicator
    new = world.restrict(odd)

    # check
    # if I have even rank
    if world.rank % 2 == 0:
        # then {new} must be {None}
        assert new is None
    # otherwise
    else:
        # I must have a valid communicator
        assert new is not None
        # whose size is related to the world size
        assert new.size == world.size // 2
        # and my ranks must be related
        assert new.rank == world.rank // 2

    return


# main
if __name__ == "__main__":
    test()


# end of file
