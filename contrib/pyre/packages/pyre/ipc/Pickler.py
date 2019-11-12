# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis, leif strand
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
import pickle
import struct
# my protocol
from . import marshaler


# class declaration
class Pickler(pyre.component, family="pyre.ipc.marshalers.pickler", implements=marshaler):
    """
    A marshaler that uses the native python services in {pickle} to serialize python objects
    for transmission to other processes.

    The {send} protocol pickles an object into the payload byte stream, and builds a header
    with the length of the payload. Similarly, {recv} first extracts the length of the byte
    string and uses that information to pull the object representation from the input
    channel. This is necessary to simplify interacting with streams that may make only portions
    of their contents available at a time.
    """


    # public data
    packing = "<L" # the struct format for encoding the payload length
    headerSize = struct.calcsize(packing)

    # interface
    @pyre.export
    def send(self, item, channel):
        """
        Pack and ship {item} over {channel}
        """
        # pickle the item
        body = pickle.dumps(item)
        # build its header
        header = struct.pack(self.packing, len(body))
        # put it together
        message = header + body
        # send it off
        return channel.write(bstr=message)


    @pyre.export
    def recv(self, channel):
        """
        Extract and return a single item from {channel}
        """
        # get the length
        header = channel.read(maxlen=self.headerSize)
        # unpack it
        length, = struct.unpack(self.packing, header)
        # get the body
        body = channel.read(minlen=length)
        # extract the object and return it
        return pickle.loads(body)


# end of file
