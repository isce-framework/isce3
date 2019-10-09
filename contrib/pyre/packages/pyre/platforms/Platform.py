# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import sys
# the framework
import pyre


# declaration
class Platform(pyre.protocol, family='pyre.platforms'):
    """
    Encapsulation of host specific information
    """


    # framework obligations
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Build the preferred host implementation
        """
        # get the platform id
        platform = sys.platform

        # if we are on darwin
        if platform.startswith('darwin'):
            # get the {Darwin} host wrapper
            from .Darwin import Darwin
            # and ask it for a suitable default implementation
            return Darwin

        # if we are on a linux derivative
        if platform.startswith('linux'):
            # get the {Linux} host wrapper
            from .Linux import Linux
            # and ask it for a suitable default implementation
            return Linux.flavor()

        # otherwise, we know nothing; let the user know
        from .Host import Host
        return Host


# end of file
