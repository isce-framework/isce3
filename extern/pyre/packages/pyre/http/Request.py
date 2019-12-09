# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re
import urllib.parse


# class declaration
class Request:
    """
    Parse and analyze an HTTP request
    """


    # exceptions
    from . import responses


    # public state
    url = None # the url the peer requested
    command = None # the type of request
    version = None # the http protocol version requested

    headers = None # dictionary of request headers
    payload = None # the request payload


    # interface
    def extract(self, server, chunk):
        """
        Process a {chunk} of bytes
        """
        # if we are still doing headers, pull them from the chunk
        offset = self.extractHeaders(server=server, chunk=chunk)
        # whatever is left is request payload
        return self.extractPayload(server=server, chunk=chunk, offset=offset)


    # implementation details
    def extractHeaders(self, server, chunk):
        """
        Extract RFC2822 headers from the bytes sent by the peer
        """
        # if i am done processing headers
        if self.described:
            # bail and indicate that no bytes from {chunk} were consumed
            return 0

        # get my header encoding
        encoding = self.HEADER_ENCODING

        # a cursor into {chunk}
        offset = 0
        # if i don't know my command yet
        if not self.command:
            # this is a brand new request
            match = self.protocol.match(chunk)
            # if it didn't match
            if not match:
                # complain
                raise self.responses.BadRequestSyntax(server=server)
            # otherwise, unpack
            command, url, major, minor = match.groups()
            # and store
            self.command = command.decode(encoding)
            self.url = urllib.parse.unquote(url.decode(encoding))
            self.version = (int(major), int(minor))
            # initialize my headers
            self.headers = {}
            # update the cursor
            offset = match.end()

        # until something happens
        while True:
            # look for a header
            match = self.keyval.match(chunk, offset)
            # if it didn't match
            if not match:
                # bail
                break
            # otherwise, unpack
            key, value = match.groups()
            # decode
            key = key.decode(encoding)
            value = value.decode(encoding)
            # store
            self.headers[key] = value
            # update the cursor
            offset = match.end()

        # the next entry must be a blank line
        match = self.blank.match(chunk, offset)
        # if it doesn't match
        if not match:
            # complain
            raise self.responses.BadRequestSyntax(server=server)
        # mark me as having processed success
        self.described = True

        # and pass on how much of {chunk} i took care of
        return match.end()


    def extractPayload(self, server, chunk, offset):
        """
        Extract a {chunk} of bytes and store them
        """
        # if i am done, i am done
        if self.complete: return True

        # compute the actual size of the unprocessed part of {chunk}
        actual = len(chunk) - offset

        # check whether
        try:
            # the client specified what my payload size is
            size = self.headers['Content-Length']
        # if not
        except (KeyError, TypeError) as error:
            # the only option is that this is the end of the chunk
            if actual == 0:
                # in which case mark me as done
                self.complete = True
                # normalize my payload, if necessary
                if self.payload is None: self.payload = []
                # and get out of here
                return True
            # otherwise, complain
            raise self.responses.BadRequestSyntax(server=server) from error
        # if all goes well
        else:
            # convert into an integer
            size = int(size)

        # initialize the storage for my payload
        if self.payload is None: self.payload = []

        # compute the total payload size, including what I have gathered so far
        for portion in self.payload:
            # by adding the length of each portion to the unprocessed {chunk}
            actual += len(portion)

        # if storing this chunk would go over the limit
        if actual > size:
            # complain
            raise self.responses.RequestEntityTooLarge(server=server)
        # otherwise, store
        self.payload.append(chunk if offset == 0 else chunk[offset:])
        # check whether this was enough bytes
        self.complete = True if actual == size else False
        # and pass this info on
        return self.complete


    # debugging support
    def dump(self, channel, indent='', showHeaders=True, showPayload=True):
        """
        Place debugging information in the given channel
        """
        # meta-data
        channel.line("{}request:".format(indent))
        channel.line("{}  type: {.command!r}".format(indent, self))
        channel.line("{}  path: {.url!r}".format(indent, self))
        channel.line("{}  version: {.version!r}".format(indent, self))

        # print the headers
        if showHeaders:
            channel.line("{}  headers:".format(indent))
            for key, value in self.headers.items():
                channel.line("{}    {}: {!r}".format(indent, key, value))
        # print the payload
        if showPayload:
            # if there is a payload
            if self.payload:
                channel.line("{}  payload:".format(indent))
                channel.line("{}    {} bytes".format(indent, len(payload)))
                channel.line(self.payload)
            # otherwise
            else:
                # let me know
                channel.line("{}  payload: none".format(indent))

        # all done
        return


    # implementation details
    # state
    described = False # am i done processing the request meta-data
    complete = False # have i received everything i expect from the client?

    # constants
    # the expected encoding of the headers
    HEADER_ENCODING = 'iso-8859-1'
    # scanners
    blank = re.compile(b"\r?\n")
    keyval = re.compile(
        br"(?P<key>[^:]+):\s+(?P<value>[^\r\n]+)" +
        b"\r?\n")
    protocol = re.compile(
        br"(?P<command>[^\s]+)" +
        br"\s+(?P<url>[^\s]+)" +
        br"(?:\s+HTTP/(?P<major>\d+).(?P<minor>\d+))" +
        b"\r?\n")


# end of file
