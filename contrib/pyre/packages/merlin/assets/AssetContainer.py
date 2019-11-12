# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Asset import Asset


# class declaration
class AssetContainer(Asset):
    """
    Base class for all merlin asset containers
    """


    # constants
    category = "asset container"


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my container
        self.contents = {}
        # all done
        return


    # implementation details
    __slots__ = 'contents',


    # debugging support
    def dump(self, indent=''):
        super().dump(indent)
        for asset in self.contents.values(): asset.dump(indent=indent+'  ')
        return


# end of file
