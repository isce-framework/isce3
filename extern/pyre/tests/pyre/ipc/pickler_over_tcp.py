#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Build two processes that communicate using pickler over a pair of sockets

The server process acquires a port, to which it listens for incoming connections; the client
connects to this port and the two exchange a couple of simple messages.

In order to inform the client about the port number, and to avoid other synchronization
problems, the test case builds a pipe between the client and the server. The client waits for
data to come over its end of the pipe. The server acquires a port, and then communicates the
port number to the client through the pipe. The client sends a simple message, which the server
receives and validates. It responds with a simple message of its own and shuts down its socket
and its pipe. The client receives its message, validates and exits.
"""

# externals
import os
# access the pyre ipc package
import pyre.ipc

def test():
    # make a pickler
    m = pyre.ipc.newPickler()
    # and a pair of pipes
    parent, child = pyre.ipc.pipe()

    # fork
    pid = os.fork()
    # in the parent process
    if pid > 0:
        # invoke the parent behavior
        return onServer(marshaler=m, pipe=parent)
    # in the child, become the client
    return onClient(marshaler=m, pipe=child)


# the greetings
hello = "hello"
goodbye = "goodbye"


def onServer(marshaler, pipe):
    """Send a simple message and wait for the response"""

    # build a port
    port = pyre.ipc.port()
    # print what it was bound to
    # print("server: established port at {!r}:{}".format(*port.address.value))
    # send it in a message to the client
    marshaler.send(item=port.address, channel=pipe)
    # and wait for an incoming connection
    peer, address = port.accept()
    # print("server: connection from {}".format(address))

    # get the message
    message = marshaler.recv(channel=peer)
    # print("server: message={!r}".format(message))
    # check it
    assert message == hello
    # say goodbye
    marshaler.send(item=goodbye, channel=peer)

    # shut everything down
    port.close()

    # all done
    return


def onClient(marshaler, pipe):
    """Wait for a message and send a response"""
    # get the port number
    address = marshaler.recv(channel=pipe)
    # print it
    # print("client: address={}".format(address))
    # make a channel
    peer = pyre.ipc.tcp(address=address)
    # send a message
    marshaler.send(item=hello, channel=peer)
    # get the response
    response = marshaler.recv(channel=peer)
    # print("client: response={!r}".format(response))
    # check it
    assert response == goodbye

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
