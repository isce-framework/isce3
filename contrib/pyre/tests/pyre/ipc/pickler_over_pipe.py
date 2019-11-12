#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Build two processes that communicate using pickler over a pair of pipes
"""


def test():
    # externals
    import os
    # access the package
    import pyre.ipc

    # make a pickler
    m = pyre.ipc.newPickler()
    # and a pair of pipes
    parent, child = pyre.ipc.pipe()

    # fork
    pid = os.fork()
    # in the parent process
    if pid > 0:
        # invoke the parent behavior
        return onParent(marshaler=m, pipe=parent)
    # in the child
    return onChild(marshaler=m, pipe=child)


# the trivial messages
hello = "hello"
goodbye = "goodbye"


def onParent(marshaler, pipe):
    """Send a simple message and wait for the response"""
    # send a message
    marshaler.send(hello, pipe)
    # get the response
    response = marshaler.recv(pipe)
    # check it
    assert response == goodbye
    # and return
    return


def onChild(marshaler, pipe):
    """Wait for a message and send a response"""
    # get the message
    message = marshaler.recv(pipe)
    # check it
    assert message == hello
    # send the response
    marshaler.send(goodbye, pipe)
    # and return
    return


# main
if __name__ == "__main__":
    test()


# end of file
