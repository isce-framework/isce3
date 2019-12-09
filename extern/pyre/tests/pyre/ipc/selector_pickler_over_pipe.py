#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise a selector watching over file descriptors
"""

# externals
import os
import pyre.ipc

# if necessary
import journal
parentdbg = journal.debug("selector.parent")
# parentdbg.active = True
childdbg = journal.debug("selector.child")
# childdbg.active = True


def test():
    # build the marshaler
    m = pyre.ipc.newPickler()
    # and the communication channels
    parent, child = pyre.ipc.pipe()

    # fork
    pid = os.fork()
    # in the parent process
    if pid > 0:
        # invoke the parent behavior
        return onParent(child_pid=pid, marshaler=m, channel=child)

    # in the child process
    return onChild(marshaler=m, channel=parent)


def onParent(child_pid, marshaler, channel):
    # observe the parent selector at work
    # journal.debug("pyre.ipc.selector").active = True

    # instantiate a selector
    parentdbg.log("parent: building a selector")
    s = pyre.ipc.newSelector()

    # write-ready handler
    def parent_send(channel, **kwds):
        """send a string to the child"""

        # register the response handler; do this early to avoid race conditions
        parentdbg.log("parent: registering the response handler")
        s.whenReadReady(channel=channel, call=parent_get)

        parentdbg.log("parent: preparing the message")
        # prepare the message
        message = "Hello {}!".format(child_pid)

        # send the message
        parentdbg.log("parent: sending the message")
        marshaler.send(item=message, channel=channel)
        parentdbg.log("parent: done sending the message")

        # and return {False} so the selector stops watching the output channel
        return False

    # read-ready handler
    def parent_get(channel, **kwds):
        """receive the response from the child"""

        parentdbg.log("parent: getting response from child")
        # get the response
        message = marshaler.recv(channel)
        parentdbg.log("message={!r}".format(message))
        # check it
        parentdbg.log("parent: checking child response")
        assert message == "Goodbye from {}!".format(child_pid)
        parentdbg.log("parent: all good")
        # and return {False} so the selector stops watching the input channel
        return False

    # let me know when my pipe TO the child is ready for writing
    parentdbg.log("parent: registering the child response handler")
    s.whenWriteReady(channel=channel, call=parent_send)
    # invoke the selector
    parentdbg.log("parent: initiating exchange")
    s.watch()
    parentdbg.log("parent: all done; exiting")
    # all done
    return


def onChild(marshaler, channel):

    # observe the child selector at work
    # journal.debug("pyre.ipc.selector").active = True

    # instantiate a selector
    childdbg.log("child: building a selector")
    s = pyre.ipc.newSelector()

    # get my pid
    child_pid = os.getpid()

    # read-read handler
    def child_get(channel, **kwds):
        """receive a message from my parent"""
        childdbg.log("child: receiving message from parent")
        message = marshaler.recv(channel)
        childdbg.log("message={!r}".format(message))
        # check it
        childdbg.log("child: checking it")
        assert message == "Hello {}!".format(child_pid)
        childdbg.log("child: all good")
        # register the response handler
        parentdbg.log("child: registering the response sender")
        s.whenWriteReady(channel=channel, call=child_send)
        # and return {False} so the selector stops watching the input channel
        return False

    def child_send(channel, **kwds):
        """send a response to my parent"""

        childdbg.log("child: preparing the response")
        # create the payload
        message = "Goodbye from {}!".format(child_pid)

        # send the message
        childdbg.log("child: sending the response")
        marshaler.send(item=message, channel=channel)
        childdbg.log("child: done sending the response")

        # and return {False} so the selector stops watching the output channel
        return False

    # let me know when my pipe FROM my parent is ready for writing
    childdbg.log("child: registering the child response handler")
    s.whenReadReady(channel=channel, call=child_get)
    # invoke the selector
    childdbg.log("child: waiting for exchange")
    s.watch()
    childdbg.log("child: all done; exiting")

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
