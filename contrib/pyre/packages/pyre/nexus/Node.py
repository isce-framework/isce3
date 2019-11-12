# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import signal, weakref
# support
import pyre
# base class
from .Peer import Peer
# my protocols
from .Nexus import Nexus
from .Service import Service


# declaration
class Node(Peer, family="pyre.nexus.servers.node", implements=Nexus):


    # user configurable state
    services = pyre.properties.dict(schema=Service())
    services.doc = 'the table of available services'


    # interface
    @pyre.export
    def prepare(self, application):
        """
        Get ready to listen for incoming connections
        """
        # save a weak reference to the application context
        self.application = weakref.proxy(application)
        # go through my services
        for name, service in self.services.items():
            # show me
            application.debug.log('{}: activating {!r}'.format(self, name))
            # and activate them
            service.activate(application=application, dispatcher=self.dispatcher)
        # all done
        return


    @pyre.export
    def shutdown(self):
        """
        Shut everything down and exit gracefully
        """
        # get the application context
        application = self.application
        # go through my services
        for name, service in self.services.items():
            # show me
            application.debug.line('shutting down {!r}'.format(name))
            # shut it down
            service.shutdown()
        # flush
        application.debug.log()
        # all done
        return super().shutdown()


    # low level event handlers
    def reload(self):
        """
        Reload the nodal configuration for a distributed application
        """
        # NYI: what does 'reload' mean? does it involve the configuration store, or just the
        # layout of the distributed application?
        return


    def signal(self, signal, frame):
        """
        An adaptor that dispatches {signal} to the registered handler
        """
        # locate the handler
        handler = self.signals[signal]
        # and invoke it
        return handler()


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # register my signal handlers
        self.signals = self.registerSignalHandlers()
        # all done
        return


    # implementation details
    # signal handling
    def newSignalIndex(self):
        """
        By default, nodes register handlers for process termination and configuration reload
        """
        # build my signal index; allow {INT} bubble up to the app by not registering a handler
        # for it
        signals = {
            # on {HUP}, reload
            signal.SIGHUP: self.reload,
            # on {TERM}, terminate
            signal.SIGTERM: self.stop,
            }
        # and return it
        return signals


    def registerSignalHandlers(self):
        """
        By default, nodes register handlers for process termination and configuration reload
        """
        # build my signal index
        signals = self.newSignalIndex()
        # register the signal demultiplexer
        for name in signals.keys():
            # as a handler for every signal in my index
            signal.signal(name, self.signal)
        # all done
        return signals


    # private data
    application = None


# end of file
