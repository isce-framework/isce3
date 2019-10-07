#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the group manipulation interface.
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

    # build two groups
    odds = whole.exclude(ranks)
    evens = whole.include(ranks)

    # compute the union of the two
    union = odds.union(evens)
    # verify that the size is right
    assert union.size == world.size

    # compute the intersection of the two
    intersection = odds.intersection(evens)
    # verify the this is an empty group
    assert intersection.isEmpty()

    # compute the difference (world - odd)
    difference = whole.difference(odds)
    # verify it is the same size as evens
    assert difference.size == evens.size

    return


# main
if __name__ == "__main__":
    test()


# end of file
