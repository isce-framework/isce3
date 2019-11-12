# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Info import Info


# declaration
class InfoFile(Info):
    """
    Base class for encapsulating leaf meta-data for filesystem entries
    """

    # constants
    marker = 'f'
    isFolder = False


    # interface
    def identify(self, explorer, **kwds):
        """
        Tell {explorer} that it is visiting a file
        """
        # dispatch
        return explorer.onFile(info=self, **kwds)


# end of file
