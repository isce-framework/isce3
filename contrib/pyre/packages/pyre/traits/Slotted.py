# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Trait import Trait


# declaration
class Slotted(Trait):
    """
    Intermediate class that knows that trait values are held by framework slots
    """


    # types
    from .SlotFactory import SlotFactory as factory


    # framework data
    isConfigurable = True # slotted traits have configurable values
    classSlot = None # the factory for class slots
    instanceSlot = None # the factory of instance slots


    # meta-methods
    def __get__(self, instance, cls):
        """
        Retrieve the value of this trait
        """
        # find out whose inventory we are supposed to access
        configurable = instance or cls
        # get the slot from the client's inventory
        slot = configurable.pyre_inventory[self]
        # compute and return its value
        return slot.value


# end of file
