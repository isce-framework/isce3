# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre

# declaration
class Shape(pyre.protocol, family="gauss.shapes"):
    """
    Protocol declarator for geometrical regions
    """

    # my default implementation
    @classmethod
    def pyre_default(cls, **kwds):
        """
        The default {Shape} implementation
        """
        # use {Ball}
        from .Ball import Ball
        return Ball

    # interface
    @pyre.provides
    def measure(self):
        """
        Compute my measure (length, area, volume, etc)
        """

    @pyre.provides
    def contains(self, points):
        """
        Filter out {points} that are on my exterior
        """


# end of file
