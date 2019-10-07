# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis, leif strand
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Port:
    """
    A mechanism that enables peer processes to draw the attention of the process owning the
    port
    """


    # interface
    @classmethod
    def open(cls, address, **kwds):
        """
        Attempt to draw the attention of the peer process at {address}
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'open'".format(type(self)))


    @classmethod
    def install(cls, **kwds):
        """
        Install and activate a port. Called by the owning process when it is ready to start
        accepting guests
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'install'".format(type(self)))


# end of file
