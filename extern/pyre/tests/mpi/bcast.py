#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise broadcast operations
"""

class message:

    def __init__(self, data, **kwds):
        super().__init__(**kwds)
        self.data = data
        return

    def __eq__(self, other):
        return self.data == other.data


def test():
    # access the package
    import mpi
    # initialize
    mpi.init()
    # get the world communicator
    world = mpi.world
    # set up a source for the broadcast
    source = int(world.size / 2)
    # create a message
    item = message(data="Hello from {}".format(source))
    # broadcast it
    received = world.bcast(item=item, source=source)
    # check it
    assert received == item
    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
