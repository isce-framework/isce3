# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre
# my protocol
from .Asynchronous import Asynchronous


# declaration
class Peer(pyre.component, family='pyre.nexus.peers.peer', implements=Asynchronous):
    """
    A component base class that supplies the two ingredients necessary for building event
    driven applications
    """

    # user configurable state
    marshaler = pyre.ipc.marshaler()
    marshaler.doc = "the serializer that enables the transmission of objects among peers"

    dispatcher = pyre.ipc.dispatcher()
    dispatcher.doc = "the manager of the event loop"


    # obligations
    @pyre.export
    def run(self):
        """
        Start processing requests
        """

        # prepare the execution context
        self.prepare()
        # start processing events
        status = self.watch()
        # when everything is done
        self.shutdown()
        # and report the status
        return status


    @pyre.export
    def prepare(self):
        """
        Carry out any necessary start up steps
        """
        # nothing to do
        return


    @pyre.export
    def watch(self):
        """
        Activate my event loop
        """
        # enter the event loop; we get out of here only when the dispatcher recognizes that
        # there is nothing else to do
        return self.dispatcher.watch()


    @pyre.export
    def shutdown(self):
        """
        Shut the peer down and exit gracefully
        """
        # no clean up, by default
        return


    @pyre.export
    def stop(self):
        """
        Signal my event loop to stop processing events
        """
        # let my event dispatcher know
        return self.dispatcher.stop()


    # meta-methods
    def __init__(self, name=None, timer=None, **kwds):
        # chain up
        super().__init__(name=name, **kwds)

        # i am not have a name, but i need one in what follows
        name = name or self.pyre_family() or "pyre.nexus.peers"

        # if i were handed a timer to use
        if timer is not None:
            # save it
            self.timer = timer
        # otherwise
        else:
            # make a new one and start it
            self.timer = self.pyre_executive.newTimer(name=name).start()

        # journal channels
        import journal
        self.info = journal.info(name=name)
        self.debug = journal.debug(name=name)
        self.warning = journal.warning(name=name)
        self.error = journal.error(name=name)

        # all done
        return


    # private data
    timer = None


# end of file
