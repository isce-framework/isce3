# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Info import Info


# declaration
class InfoFolder(Info):
    """
    Base class for encapsulating container meta-data for filesystem entries
    """

    # constants
    marker = 'd'
    isFolder = True


    # interface
    def identify(self, explorer, **kwds):
        """
        Tell {explorer} that it is visiting a folder
        """
        # dispatch
        return explorer.onFolder(info=self, **kwds)


# end of file
