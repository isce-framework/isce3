# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .File import File


# class declaration
class BlockDevice(File):
    """
    Representation of block devices, a type of unix device driver
    """

    # constant
    marker = 'b'


    # interface
    def identify(self, explorer, **kwds):
        """
        Tell {explorer} that it is visiting a block device
        """
        # dispatch
        return explorer.onBlockDevice(info=self, **kwds)


# end of file
