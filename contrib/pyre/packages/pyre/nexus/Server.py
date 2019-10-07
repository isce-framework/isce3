# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
import weakref
# my protocol
from .Service import Service


# my declaration
class Server(pyre.component, implements=Service):
    """
    The base class for network servers
    """


    # user configurable state
    address = pyre.properties.inet()


    # types
    from .exceptions import ConnectionResetError


    # behaviors
    @pyre.export(tip='register this service with the nexus')
    def activate(self, application, dispatcher):
        """
        Grab a port and register it with the event dispatcher
        """
        # save the application context
        self.application = weakref.proxy(application)
        # and the dispatcher
        self.dispatcher = weakref.proxy(dispatcher)
        # build a port
        port = pyre.ipc.port(address=self.address)
        # adjust my address
        self.address = port.address
        # ask the application dispatcher to monitor my port
        dispatcher.whenReadReady(channel=port, call=self.acknowledge)
        # all done
        return


    @pyre.export(tip='acknowledge a peer that has initiated a connection')
    def acknowledge(self, channel):
        """
        A peer has attempted to establish a connection
        """
        # accept the connection
        newChannel, peerAddress = channel.accept()
        # log the request
        self.application.debug.log(
            "{}: received 'connection' request from {}".format(channel, peerAddress))

        # if this is not a valid connection
        if not self.validate(channel=newChannel, address=peerAddress):
            # get rid of it
            newChannel.close()
            # and bail; indicate that i am interested in acknowledging other requests
            return True

        # process the connection; reschedule this handler to process more connection attempts
        # if {newPeer} returns {True}
        return self.connect(channel=newChannel, address=peerAddress)


    @pyre.export(tip='determine whether to start a conversation with the peer')
    def validate(self, channel, address):
        """
        Examine the peer {address} and determine whether to continue the conversation
        """
        # be friendly, by default
        return True


    @pyre.export(tip='prepare to accept connections from peers')
    def connect(self, channel, address):
        """
        Prepare to start accepting requests from peers
        """
        # N.B.: override this to implement a different connection handling strategy, i.e. fork
        # or spawn a thread; this implementation just adds the {channel} to the read pile and
        # processes client requests in the same process space

        # get the dispatcher
        dispatcher = self.dispatcher
        # place the channel on the read list
        dispatcher.whenReadReady(channel=channel, call=self.process)
        # indicate that i would like to continue receiving connection requests from other peers
        return True


    @pyre.export(tip='process and respond to the peer request')
    def process(self, channel):
        """
        Say something to the peer
        """
        # don't have anything useful to do here, so just close the connection
        channel.close()
        # and prevent this from getting rescheduled; this is bad behavior because it can
        # potentially leave data in the channel, and it ignores the event raised when the peer
        # closes the connection. subclasses should override this and not chain
        return False


    @pyre.export(tip='shutdown')
    def shutdown(self):
        """
        Clean up and shutdown
        """
        # remove my references to the application and the dispatcher
        del self.dispatcher
        del self.application
        # not much to do
        return


    # implementation details
    # private data
    application = None
    dispatcher = None


# end of file
