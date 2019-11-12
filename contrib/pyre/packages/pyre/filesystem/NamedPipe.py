# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .File import File

# class declaration
class NamedPipe(File):
    """
    Representation of named pipes, a unix interprocess communication mechanism
    """

    # constant
    marker = 'p'


    # interface
    def identify(self, explorer, **kwds):
        """
        Tell {explorer} that it is visiting a FIFO
        """
        # dispatch
        return explorer.onNamedPipe(info=self, **kwds)


# end of file
