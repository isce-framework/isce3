# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# my superclass
from .Status import Status


# declaration
class FactoryStatus(Status):
    """
    A helper that watches over the traits of factories and records value changes
    """


    # interface
    def addOutputBinding(self, factory, product):
        """
        Add {product} as an output of my {factory}
        """
        # add the {product} monitor to the pile of my observers
        self.addObserver(observer=product.pyre_status)
        # and chain up
        return super().addOutputBinding(factory=factory, product=product)


    def removeOutputBinding(self, factory, product):
        """
        Remove {product} as an output of my {factory}
        """
        # remove the {product} monitor from my pile of observers
        self.removeObserver(observer=product.pyre_status)
        # and chain up
        return super().removeOutputBinding(factory=factory, product=product)


# end of file
