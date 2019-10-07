# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
import collections
# access the framework
import pyre
# get my protocol
from .Terminal import Terminal as terminal


# declaration
class Plain(pyre.component, family='pyre.terminals.plain', implements=terminal):
    """
    A terminal that provides no color capabilities
    """

    # public data
    @property
    def width(self):
        """
        Compute the width of the terminal
        """
        # attempt to
        try:
            # ask python
            return os.get_terminal_size().columns
        # if something went wrong
        except OSError:
            # absorb
            pass
        # don't know
        return 0


    # interface
    def rgb(self, **kwds):
        """
        The 24-bit color request
        """
        # we don't do this...
        return''


    def rgb256(self, red=0, green=0, blue=0, foreground=True):
        """
        The 256-color palette request
        """
        # we don't do this either...
        return''


    # implementation details
    ansi = collections.defaultdict(str) # all color decorations are empty strings...
    x11 = collections.defaultdict(str) # all color decorations are empty strings...
    gray = collections.defaultdict(str) # all color decorations are empty strings...
    misc = collections.defaultdict(str) # all color decorations are empty strings...


# end of file
