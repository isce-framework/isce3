# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import weakref
# support
import pyre
# my protocol
from .Specification import Specification
# my superclass
from .Node import Node


# class declaration
class Product(Node, implements=Specification, internal=True):
    """
    The base class for data products
    """


    # protocol obligations
    @pyre.export
    def pyre_make(self, **kwds):
        """
        Invoke my factories to update me
        """
        # if i am not stale
        if self.pyre_stale is False:
            # nothing to do
            return self
        # otherwise, go through my factories
        for factory in self.pyre_factories:
            # and ask each one to update me
            factory.pyre_make(**kwds)
        # if all went well, update my status
        self.pyre_stale = False
        # all done
        return self


    @pyre.export
    def pyre_tasklist(self, **kwds):
        """
        Generate the sequence of factories that must be invoked to rebuild me
        """
        # if i'm not stale
        if self.pyre_stale is False:
            # nothing to do
            return
        # otherwise, go through my factories
        for factory in self.pyre_factories:
            # and ask them for their contribution
            yield from factory.pyre_tasklist(**kwds)
        # all done
        return


    @pyre.export
    def pyre_targets(self, **kwds):
        """
        Generate the sequence of products that must be refreshed to rebuild me
        """
        # if i'm not stale
        if self.pyre_stale is False:
            # nothing to do
            return
        # otherwise, go through my factories
        for factory in self.pyre_factories:
            # ask them for their contribution
            yield from factory.pyre_targets(**kwds)
        # and add myself to the pile
        yield self
        # all done
        return


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize the list of my factories
        self.pyre_factories = weakref.WeakSet()
        # all done
        return


    # flow hooks
    def pyre_newStatus(self, **kwds):
        """
        Build a handler for my status changes
        """
        # grab the factory
        from .ProductStatus import ProductStatus
        # make one and return it
        return ProductStatus(**kwds)


    def pyre_addInputBinding(self, factory):
        """
        Bind me as an input to the given {factory}
        """
        # let my monitor know there is a new client {factory}
        self.pyre_status.addInputBinding(factory=factory, product=self)
        # all done
        return


    def pyre_removeInputBinding(self, factory):
        """
        Unbind me as an input to the given {factory}
        """
        # let my monitor know {factory} is no longer a client
        self.pyre_status.removeInputBinding(factory=factory, product=self)
        # all done
        return


    def pyre_addOutputBinding(self, factory):
        """
        Bind me as an output to the given {factory}
        """
        # add {factory} to my pile
        self.pyre_factories.add(factory)
        # let my monitor know there is a new client {factory}
        self.pyre_status.addOutputBinding(factory=factory, product=self)
        # all done
        return


    def pyre_removeOutputBinding(self, factory):
        """
        Unbind me as an output to the given {factory}
        """
        # remove {factory} from my pile
        self.pyre_factories.remove(factory)
        # let my monitor know {factory} is no longer a client
        self.pyre_status.removeOutputBinding(factory=factory, product=self)
        # all done
        return


# end of file
