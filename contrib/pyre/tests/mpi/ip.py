#!/usr/bin/env python3.3
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
    # externals
    import mpi
    import socket

    # get the world communicator
    world = mpi.world
    # get my ip address
    host = socket.gethostname()

    print("{0.rank:03}/{0.size:03}: {1}".format(world, host))

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
