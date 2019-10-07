# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis, leif strand
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre


# declaration
class Marshaler(pyre.protocol, family="pyre.ipc.marshalers"):
    """
    Protocol for components responsible for serializing python objects for transmission to
    other processes
    """


    # factory for my default implementation
    @classmethod
    def pyre_default(cls, **kwds):
        from .Pickler import Pickler
        return Pickler


    # interface
    @pyre.provides
    def recv(self, channel):
        """
        Extract and return one object from {channel}
        """


    @pyre.provides
    def send(self, item, channel):
        """
        Pack and ship {item} over {channel}
        """


# end of file
