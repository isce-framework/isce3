# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
# my superclass
from .Dashboard import Dashboard as dashboard


# declaration
class Component(pyre.component, dashboard, hidden=True):
    """
    Minor merlin specific embellishment of the {pyre.component} base class
    """


    # public data
    @property
    def vfs(self):
        """
        Convenient access to the application fileserver
        """
        # merlin knows
        return self.merlin.vfs


# end of file
