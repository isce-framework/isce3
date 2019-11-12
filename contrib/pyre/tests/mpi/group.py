#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that communicator groups are accessible
"""


def test():
    # access the package
    import mpi
    # initialize mpi
    mpi.init()
    # grab the world communicator
    world = mpi.world
    # access the world process group
    whole = world.group()

    # check that I can compute ranks correctly
    assert world.rank == whole.rank

    return


# main
if __name__ == "__main__":
    test()


# end of file
