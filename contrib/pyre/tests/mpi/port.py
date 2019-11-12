#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise message ports: place processors on a ring and exchange messages
"""


def test():
    # access the package
    import mpi
    # initialize
    mpi.init()
    # get the world communicator
    world = mpi.world
    # its size
    size = world.size
    # and my rank
    rank = world.rank

    # check that the world has at least two tasks
    if size < 2: return

    # the source of the message i will receive
    source = world.port(peer=(rank-1)%size)
    # the destination of the message I will send
    destination = world.port(peer=(rank+1)%size)

    # send my message to the guy to my right
    destination.sendString("Hello {}!".format(destination.peer))
    # and receive a message from the guy to my left
    message = source.recvString()
    # and check its contents
    assert message == "Hello {}!".format(rank)

    # repeat by exchanging pickled objects
    destination.send("Hello {}!".format(destination.peer))
    message = source.recv()
    # checks
    assert message == "Hello {}!".format(rank)

    # all done
    return


# main
if __name__ == "__main__":
    # import journal
    # journal.debug("mpi.ports").active = True
    test()


# end of file
