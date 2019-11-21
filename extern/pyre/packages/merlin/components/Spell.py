# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# framework access
import pyre
# my superclass
from .Dashboard import Dashboard
# my protocol
from .Action import Action


# class declaration
class Spell(pyre.panel(), Dashboard):
    """
    Base class for merlin spell implementations
    """


    # public data
    @property
    def vfs(self):
        """
        Access to the fileserver
        """
        # merlin knows
        return self.merlin.vfs


# end of file
