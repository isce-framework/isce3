# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis, leif strand
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import socket
# my interface
from .Socket import Socket


# declaration
class SocketTCP(Socket):
    """
    A channel that uses TCP sockets as the communication mechanism
    """


    # constants
    type = socket.SOCK_STREAM


    # input/output
    def read(self, minlen=0, maxlen=4*1024):
        """
        Read {count} bytes from my input channel
        """
        # adjust the inputs
        if maxlen < minlen: maxlen = minlen
        # reset the byte count
        total = 0
        # initialize the packet pile
        packets = []
        # for as long as it takes
        while True:
            # pull something from the channel
            packet = self.recv(maxlen-total)
            # get its length
            got = len(packet)
            # if we got nothing, the channel is closed; bail
            if got == 0: break
            # otherwise, update the total
            total += got
            # and save the packet
            packets.append(packet)
            # if we have reached our goal, bail
            if total >= minlen: break
        # assemble the byte string and return it
        return b''.join(packets)


    def write(self, bstr):
        """
        Write the bytes in {bstr} to my output channel
        """
        # make sure the entire byte string is delivered
        self.sendall(bstr)
        # and return the number of bytes sent
        return len(bstr)


    # meta-methods
    def __str__(self):
        return "tcp socket to {.peer}".format(self)


    # implementation details
    __slots__ = () # socket has it, so why not...


# end of file
