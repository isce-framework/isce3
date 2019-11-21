# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre


# my declaration
class Server(pyre.nexus.server, family='pyre.nexus.servers.http'):
    """
    A server that understands HTTP
    """


    # types
    from .Request import Request as request
    from .Response import Response as response
    # exceptions
    from . import exceptions, responses, documents


    # user configurable state
    renderer = pyre.weaver.language(default='http')
    renderer.doc = 'the renderer of the server responses to client requests'


    # public state
    @property
    def name(self):
        """
        Server identification
        """
        # build an identification string
        return "pyre/{}.{}.{}".format(*self.version)


    @property
    def version(self):
        """
        Server version
        """
        # inherit the version of the framework
        return pyre.version()


    # protocol obligations
    @pyre.export(tip='respond to the peer request')
    def process(self, channel):
        """
        Initiate or continue a conversation with a peer over {channel}
        """
        # we are here because the {channel} has data for me; one of the following is true
        #
        # - this is the notification that the connection has been closed by the peer; in this
        #   case the channel contains exactly zero bytes for us; we check for this one early
        # - this the first time this peer connects
        # - more data that for an existing request have arrived
        # - this is a known peer whose previous request was handled but has kept the connection
        #   alive; this case may not survive as it can be handled by starting a new request
        #   every time without closing the connection
        # - the client has pipelined its requests; currently, this case is not handled
        #   correctly because it involves saving the local buffers

        # get the application context
        application = self.application
        # show me
        application.debug.log('reading data from {}'.format(channel.peer))
        # get whatever data is available at this point
        chunk = channel.read(maxlen=self.MAX_BYTES)

        # if there was nothing to read
        if len(chunk) == 0:
            # show me
            application.debug.log('connection from {} was closed'.format(channel.peer))
            # close the connection
            channel.close()
            # check whether we know this peer
            try:
                # in which case, forget him
                del self.requests[channel]
            # if not
            except KeyError:
                # no problem
                pass
            # stop listening
            return False

        # well, there is data to process; if we have a pending request
        try:
            # get it
            request = self.requests[channel]
        # otherwise
        except KeyError:
            # make a new one
            request = self.request()

        # attempt to
        try:
            # hand the chunk to the request
            complete = request.extract(server=self, chunk=chunk)
        # if something wrong happened
        except self.exceptions.ProtocolError as error:
            # send an error report to the client
            return self.respond(channel=channel, response=error)

        # if request assembly is not finished yet
        if not complete:
            # we expect more data to arrive later; register this request so we can continue the
            # processing next time there are data for it
            self.requests[channel] = request
            # and reschedule this channel
            return True

        # figuring out what the client is asking for is now complete; try to
        try:
            # fulfill the request
            response = self.fulfill(request)
        # if something bad happened
        except self.exceptions.ProtocolError as error:
            # send an error report to the client
            return self.respond(channel=channel, response=error)

        # if all goes well, respond
        return self.respond(channel=channel, response=response)


    # interface
    def fulfill(self, request):
        """
        Fulfill the given fully formed client {request}
        """
        # delegate to the app to build a response a message and return it
        return self.application.pyre_respond(server=self, request=request)


    def respond(self, channel, response):
        # attempt to
        try:
            # ask the renderer to put together the byte stream
            stream = b'\r\n'.join(self.renderer.render(server=self, document=response))
        # if something goes wrong
        except self.exceptions.ProtocolError as error:
            # render the error
            stream = b'\r\n'.join(self.renderer.render(server=self, document=error))

        # either way, send the bytes to the client
        channel.write(stream)
        # keep the channel alive
        return True


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my connectionn index
        self.requests = {}
        # all done
        return


    # implementation details
    # private data
    requests = None
    # constants
    MAX_BYTES = 1024 * 1024


# end of file
