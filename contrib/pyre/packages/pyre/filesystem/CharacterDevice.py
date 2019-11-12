# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .File import File

# class declaration
class CharacterDevice(File):
    """
    Representation of character devices, a type of unix device driver
    """

    # constant
    marker = 'c'


    # interface
    def identify(self, explorer, **kwds):
        """
        Tell {explorer} that it is visiting a serial device
        """
        # dispatch
        return explorer.onCharacterDevice(info=self, **kwds)


# end of file
