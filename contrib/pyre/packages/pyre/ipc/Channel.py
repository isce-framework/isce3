# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis, leif strand
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Channel:
    """
    A wrapper around the lower level IPC mechanisms that normalizes the sending and receiving
    of messages. See {Pipe} and {Socket} for concrete examples of encapsulation of the
    operating system services.
    """


    # interface
    # channel life cycle management
    @classmethod
    def open(cls, **kwds):
        """
        Channel factory
        """
        raise NotImplementedError("class {.__name__!r} must implement 'open'".format(cls))


    def close(self):
        """
        Shutdown the channel
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'close'".format(type(self)))

    # access to the individual channel end points
    @property
    def inbound(self):
        """
        Retrieve the channel end point that can be read
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'inbound'".format(type(self)))

    @property
    def outbound(self):
        """
        Retrieve the channel end point that can be written
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'outbound'".format(type(self)))

    # input/output
    def read(self, minlen, maxlen):
        """
        Read up to {count} bytes from my input channel
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'read'".format(type(self)))


    def write(self, bstr):
        """
        Write the bytes in {bstr} to output channel
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'write'".format(type(self)))


# end of file
