# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from ..calc.NodeInfo import NodeInfo


# declaration
class SlotInfo(NodeInfo):
    """
    Encapsulation of the slot metadata maintained by the nameserver
    """


    # types
    from .Priority import Priority as priorities
    from ..traits.Property  import Property as properties


    # public data
    locator = None # provenance
    priority = None # the rank of this setting
    factory = None # the type information


    # meta-methods
    def __init__(self, priority=None, locator=None, factory=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my metadata
        self.locator = locator
        self.priority = priority or self.priorities.uninitialized()
        self.factory = factory or self.properties.identity(name=self.name).instanceSlot
        # all done
        return


# end of file
