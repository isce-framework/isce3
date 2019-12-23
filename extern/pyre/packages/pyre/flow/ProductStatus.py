# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# my superclasses
from .Status import Status


# declaration
class ProductStatus(Status):
    """
    A helper that watches over the traits of products and records value changes
    """

    # interface
    def addInputBinding(self, factory, product):
        """
        My client {product} is an input to {factory}
        """
        # add the {factory} monitor to my observers
        self.addObserver(observer=factory.pyre_status)
        # and chain up
        return super().addInputBinding(factory=factory, product=product)


    def removeInputBinding(self, factory, product):
        """
        My client {product} is no longer an input to {factory}
        """
        # remove the {factory} monitor from my pile of observers
        self.removeObserver(observer=factory.pyre_status)
        # and chain up
        return super().removeInputBinding(factory=factory, product=product)


    def addOutputBinding(self, factory, product):
        """
        Add my client {product} as an output of {factory}
        """
        # my client is associated with a new factory, so mark me as stale and notify downstream
        self.flush(observable=factory.pyre_status)
        # and chain up
        return super().addOutputBinding(factory=factory, product=product)


# end of file
