# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis, leif strand
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import socket
# my interface
from .Channel import Channel


# declaration
class Socket(socket.socket, Channel):
    """
    A channel that uses sockets as the communication mechanism

    This class captures the part of the {socket} interface that is independent of the type of
    socket. The implementation of the remainder of the {Channel} interface is provided by
    subclasses.
    """


    # types
    from ..schemata import inet


    # access to the individual channel end points
    @property
    def inbound(self):
        """
        Retrieve the channel end point that can be read
        """
        # easy enough
        return self


    @property
    def outbound(self):
        """
        Retrieve the channel end point that can be written
        """
        # easy enough
        return self


    # access to the socket properties
    @property
    def peer(self):
        """
        Return the address of my peer, i.e. the remote endpoint of the socket
        """
        # get the raw address
        address = self.getpeername()
        # parse it, decorate it and return it
        return self.inet().recognize(family=self.family, address=address)


    # interface
    def accept(self):
        """
        Wait for a connection attempt, build a channel around the socket to the peer, and
        return it along with the address of the remote process
        """
        # bypass the socket interface because it calls the wrong constructor explicitly
        fd, address = self._accept()
        # build the channel
        channel = type(self)(self.family, self.type, self.proto, fileno=fd)
        # build the address
        address = self.inet().recognize(family=self.family, address=address)
        # adjust the socket flags; see {socket.py} in the standard library for more details
        if socket.getdefaulttimeout() is None and self.gettimeout(): channel.setblocking(True)
        # return the channel to and the address of the peer process
        return channel, address


    # meta-methods
    def __str__(self):
        return "socket to {.peer}".format(self)


    # implementation details
    __slots__ = () # socket has it, so why not...


# end of file
