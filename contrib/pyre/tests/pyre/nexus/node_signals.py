#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the node base class handles signals properly
"""

# externals
import os, sys, signal
# framework parts
import pyre, journal


# the launcher
def test():
    # N.B.: testing the delivery of signals is a bit tricky. {fork} is not sufficient: there
    # must be an {exec} as well, otherwise signals do not get delivered properly

    # grab the nameserver
    ns = pyre.executive.nameserver

    # if this is the fork/exec child
    if 'child' in ns:
        # spit out the command line
        # print("child:", sys.argv)
        # grab the file descriptors
        infd = int(ns["infd"])
        outfd = int(ns["outfd"])
        # convert them into a channel
        channel = pyre.ipc.pipe(descriptors=(infd, outfd))
        # invoke the child behavior
        return onChild(channel=channel)

    # otherwise, set the parent/child process context
    # build the communication channels
    parent, child = pyre.ipc.pipe()

    # fork
    pid = os.fork()
    # in the parent process
    if pid > 0:
        # invoke the parent behavior
        return onParent(childpid=pid, channel=parent)

    # in the child process, build the new command line
    argv = [sys.executable] + sys.argv + [
        "--child",
        "--infd={}".format(child.infd),
        "--outfd={}".format(child.outfd)
        ]
    # print("execv:", argv)

    # on python 3.4 and later
    try:
        # we have to explicitly ask for the pipes to become available across the call to
        # {exec}; grab the function that enables file descriptor inheritance
        inherit = os.set_inheritable
    # older versions don't have this function
    except AttributeError:
        # but also allow subprocesses to inherit file descriptors, so no worries
        pass
    # if all is well
    else:
        # mark the two descriptors
        inherit(child.infd, True)
        inherit(child.outfd, True)

    # and exec
    return os.execv(sys.executable, argv)


# the parent behavior
def onParent(childpid, channel):
    """
    The parent waits until the pipe to the child is ready for writing and then sends a SIGHUP
    to the child. The child is supposed to respond by writing 'reloaded' to the pipe, so the
    parent schedules a handler to receive the message and respond by issuing a SIGTERM, which
    kills the child. The parent harvests the exit status, checks it and terminates.
    """

    # debug
    pdbg = journal.debug("parent")
    # log
    pdbg.log("in the parent process")

    # base class
    from pyre.nexus.Node import Node
    # subclass Node
    class node(Node):

        marshaler = pyre.ipc.marshaler()

        def recvReady(self, **kwds):
            # log
            pdbg.log("receiving message from child")
            # receive
            message = self.marshaler.recv(channel=self.channel)
            # log and check
            pdbg.log("child said {!r}".format(message))
            assert message == 'ready'
            # register the handler for the response to 'reload'
            self.dispatcher.whenReadReady(channel=self.channel, call=self.recvReloaded)
            # issue the 'reload' signal
            os.kill(childpid, signal.SIGHUP)
            # don't reschedule this handler
            return False

        def recvReloaded(self, **kwds):
            """check the response to 'reload' and send 'terminate'"""
            # log
            pdbg.log("receiving message from child")
            # receive
            message = self.marshaler.recv(channel)
            # check it
            pdbg.log("child said {!r}".format(message))
            assert message == "reloaded"
            # now, send a 'terminate' to my child
            pdbg.log("sending 'terminate'")
            os.kill(childpid, signal.SIGTERM)
            # and stop my dispatcher
            self.dispatcher.stop()
            pdbg.log("all good")
            # don't reschedule this handler
            return False

        def __init__(self, channel, **kwds):
            # chain up
            super().__init__(**kwds)
            # save my channel
            self.channel = channel
            # show me
            pdbg.log("registering 'recvReady'")
            # register my handler
            self.dispatcher.whenReadReady(channel=channel, call=self.recvReady)
            # all done
            return

    # create a node
    pdbg.log("instantiating my node")
    parent = node(name="parent", channel=channel)

    # register my 'sendReload' to be invoked when the pipe is channel is ready for write
    # watch for events
    pdbg.log("entering event loop")
    parent.dispatcher.watch()
    # wait for it to die
    pdbg.log("waiting for child to die")
    pid, status = os.wait()
    # check the pid
    pdbg.log("checking pid")
    assert pid == childpid
    # check the status
    code = (status & 0xF0)
    reason = status & 0x0F
    pdbg.log("checking the status: code={}, reason={}".format(code, reason))
    assert code == 0 and reason == 0
    pdbg.log("exiting")

    # all done
    return parent


# the child behavior
def onChild(channel):
    """
    The child enters an indefinite loop by repeatedly scheduling an alarm. The modified {Node}
    overrides the 'reload' signal handler to send an acknowledgment to the parent, and goes
    back to its indefinite loop. Eventually, the parent sends a SIGTERM, which kills the child
    """

    # debug
    cdbg = journal.debug("child")
    # log
    cdbg.log("in the child process: channel={}".format(channel))

    # base class
    from pyre.nexus.Node import Node
    # subclass Node
    class node(Node):

        marshaler = pyre.ipc.marshaler()

        def sendReady(self, **kwds):
            # log
            cdbg.log("sending 'ready'")
            # get it
            self.marshaler.send(item='ready', channel=self.channel)
            # don't reschedule this handler
            return False

        def sendReloaded(self, **kwds):
            # show me
            cdbg.log("sending 'reloaded' to my parent")
            # send a message to the parent
            self.marshaler.send(item="reloaded", channel=self.channel)
            # don't reschedule
            return False

        def alarm(self, timestamp):
            # show me
            cdbg.log("timeout on {}".format(timestamp))
            # raise again after 10 seconds
            return 1*self.dispatcher.second

        def reload(self):
            # show me
            cdbg.log("schedule 'sendReloaded'")
            # schedule to send a message to the parent
            self.dispatcher.whenWriteReady(channel=self.channel, call=self.sendReloaded)
            # all done
            return

        @pyre.export
        def stop(self):
            # show me
            cdbg.log("marking clean exit and stopping the dispatcher")
            # mark me
            self.cleanExit = True
            # chain up
            return super().stop()

        def __init__(self, channel, **kwds):
            # chain up
            super().__init__(**kwds)
            # show me
            cdbg.log("dispatcher: {}".format(self.dispatcher))
            # my communication channel
            self.channel = channel
            # marker that my {stop} was called
            self.cleanExit = False
            # set up an alarm to keep the process alive
            self.dispatcher.alarm(interval=1*self.dispatcher.second, call=self.alarm)
            # let my parent know I am ready
            self.dispatcher.whenWriteReady(channel=channel, call=self.sendReady)
            # all done
            return

    # instantiate
    cdbg.log("instantiating my node")
    child = node(name="child", channel=channel)
    cdbg.log("child: {}".format(child))
    # enter the event loop
    cdbg.log("entering the event loop")
    child.dispatcher.watch()
    # check that this is a clean exit
    assert child.cleanExit
    # show me
    cdbg.log("exiting")
    # return it
    return child


# main
if __name__ == "__main__":
    # progress logging
    journal.debug("child").active = False
    journal.debug("parent").active = False
    journal.debug("pyre.ipc.selector").active = False
    # do...
    test()


# end of file
