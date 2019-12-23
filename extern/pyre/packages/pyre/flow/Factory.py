# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre
# my protocol
from .Producer import Producer
# my superclass
from .Node import Node
# my meta-class
from .FactoryMaker import FactoryMaker


# class declaration
class Factory(Node, metaclass=FactoryMaker, implements=Producer, internal=True):
    """
    The base class for creators of data products
    """


    # types
    from .exceptions import IncompleteFlowError


    # protocol obligations
    @pyre.export
    def pyre_make(self, **kwds):
        """
        Construct my products
        """
        # sort my inputs
        unbound, stale, _ = self.pyre_classifyInputs()
        # if there are unbound traits
        if unbound:
            # build a locator that blames my caller
            locator = pyre.tracking.here(level=1)
            # complain
            raise self.IncompleteFlowError(node=self, traits=unbound, locator=locator)
        # go through the stale products
        for product in stale:
            # refresh them
            product.pyre_make(**kwds)
        # if anything were stale
        if stale:
            # invoke me
            self.pyre_run(stale=stale, **kwds)
        # all done
        return


    @pyre.export
    def pyre_tasklist(self, **kwds):
        """
        Generate the sequence of factories that must be invoked to rebuild a product
        """
        # sort my inputs
        unbound, stale, _ = self.pyre_classifyInputs()
        # if there are unbound traits
        if unbound:
            # build a locator that blames my caller
            locator = pyre.tracking.here(level=1)
            # complain
            raise self.IncompleteFlowError(node=self, traits=unbound, locator=locator)
        # go through the stale products
        for product in stale:
            # ask them to contribute
            yield from product.pyre_tasklist(**kwds)
        # add me to the pile
        yield self
        # all done
        return


    @pyre.export
    def pyre_targets(self, context=None):
        """
        Generate the sequence of products that must be refreshed
        """
        # sort my inputs
        unbound, stale, _ = self.pyre_classifyInputs()
        # if there are unbound traits
        if unbound:
            # build a locator that blames my caller
            locator = pyre.tracking.here(level=1)
            # complain
            raise self.IncompleteFlowError(node=self, traits=unbound, locator=locator)
        # go through the stale inputs
        for product in stale:
            # ask them to contribute
            yield from product.pyre_targets()
        # all done
        return


    # interface
    def pyre_inputs(self):
        """
        Generate the sequence of my input products
        """
        # grab my inventory
        inventory = self.pyre_inventory
        # go through my input traits
        for trait in self.pyre_inputTraits:
            # get the associated product
            product = inventory[trait].value
            # pass it on along with me as its factory, and the trait meta-data
            # N.B. the strangely articulated meta-data form is meant to accommodate workflows
            yield product, [(self, trait)]
        # all done
        return


    def pyre_outputs(self):
        """
        Generate the sequence of my output products
        """
        # grab my inventory
        inventory = self.pyre_inventory
        # go through my output traits
        for trait in self.pyre_outputTraits:
            # get the associated product
            product = inventory[trait].value
            # pass it on along with me as its factory, and the trait meta-data
            # N.B. the strangely articulated meta-data form is meant to accommodate workflows
            yield product, [(self, trait)]
        # all done
        return


    def pyre_factories(self):
        """
        Generate the sequence of my factories
        """
        # there is only me
        yield self
        # all done
        return


    def pyre_closed(self):
        """
        A factory is closed when it has no unbound products
        """
        # sort my inputs
        unboundInputs, _, _ = self.pyre_classifyInputs()
        # if any are unbound
        if unboundInputs:
            # oops
            return False
        # classify my outputs
        unboundOutputs, _ = self.pyre_classifyOutputs()
        # if any are unbound
        if unboundOutputs:
            # oops again
            return False
        # otherwise, we are good
        return True


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # get my inventory
        inventory = self.pyre_inventory
        # get my inputs
        inputs = (inventory[trait].value for trait in self.pyre_inputTraits)
        # bind me to them
        self.pyre_bindInputs(*inputs)
        # get my outputs
        outputs = (inventory[trait].value for trait in self.pyre_outputTraits)
        # bind me to them
        self.pyre_bindOutputs(*outputs)
        # all done
        return


    # flow hooks
    # deployment
    def pyre_run(self, **kwds):
        """
        Invoke me and remake my products
        """
        # nothing to do
        return self


    # status management
    def pyre_newStatus(self, **kwds):
        """
        Build a handler for my status changes
        """
        # grab the factory
        from .FactoryStatus import FactoryStatus
        # make one and return it
        return FactoryStatus(**kwds)


    # introspection
    def pyre_classifyInputs(self):
        """
        Go through my inputs and sort them in three piles: unbound, stale, and fresh
        """
        # make a pile of fresh inputs
        fresh = []
        # one for stale inputs
        stale = []
        # and another for the unbound traits
        unbound = []
        # go through my inputs
        for product, meta in self.pyre_inputs():
            # if the product is unbound
            if product is None:
                # add its meta-data to the pile
                unbound.append(meta)
                # and move on
                continue
            # if the product is stale
            if product.pyre_stale is True:
                # add it to the stale pile
                stale.append(product)
                # and move on
                continue
            # otherwise, it must be fresh
            fresh.append(product)
        # return the three piles
        return unbound, stale, fresh


    def pyre_classifyOutputs(self):
        """
        Go through my outputs and sort them into two piles: unbound and bound
        """
        # make a pile for the bound outputs
        bound = []
        # and another for the unbound ones
        unbound = []
        # go through my outputs
        for product, meta in self.pyre_outputs():
            # if the product is unbound
            if product is None:
                # add it to the pile
                unbound.append(meta)
                # and move on
                continue
            # otherwise, it's good
            bound.append(product)
        # all done
        return unbound, bound


    # connectivity maintenance
    def pyre_bindInputs(self, *inputs):
        """
        Bind me to the sequence of products in {inputs}
        """
        # get my status monitor
        monitor = self.pyre_status
        # go through each of my inputs
        for product in inputs:
            # if this is an unbound product
            if product is None:
                # skip it
                continue
            # tell the product i'm interested in its state
            product.pyre_addInputBinding(factory=self)
            # and notify my monitor
            monitor.addInputBinding(factory=self, product=product)
        # all done
        return self


    def pyre_unbindInputs(self, *inputs):
        """
        Unbind me to the sequence of products in {inputs}
        """
        # get my status monitor
        monitor = self.pyre_status
        # go through each of my inputs
        for product in inputs:
            # if this is an unbound product
            if product is None:
                # skip it
                continue
            # tell the product i'm interested in its state
            product.pyre_removeInputBinding(factory=self)
            # and notify my monitor
            monitor.removeInputBinding(factory=self, product=product)
        # all done
        return self


    def pyre_bindOutputs(self, *outputs):
        """
        Bind me to the sequence of products in {outputs}
        """
        # get my status monitor
        monitor = self.pyre_status
        # go through the products
        for product in outputs:
            # if this is an unbound product
            if product is None:
                # skip it
                continue
            # tell the product i'm its factory
            product.pyre_addOutputBinding(factory=self)
            # and notify my monitor
            monitor.addOutputBinding(factory=self, product=product)
        # all done
        return self


    def pyre_unbindOutputs(self, *outputs):
        """
        Unbind me to the sequence of products in {outputs}
        """
        # get my status monitor
        monitor = self.pyre_status
        # go through the products
        for product in outputs:
            # if this is an unbound product
            if product is None:
                # skip it
                continue
            # tell the product i'm not its factory any more
            product.pyre_removeOutputBinding(factory=self)
            # and notify my monitor
            monitor.removeOutputBinding(factory=self, product=product)
        # all done
        return self


    # framework hooks
    def pyre_traitModified(self, trait, new, old):
        """
        Hook invoked when a trait changes value
        """
        # evaluate
        newValue = new.value
        oldValue = old.value
        # if {trait} is an input
        if trait.input:
            # remove from my input pile
            self.pyre_unbindInputs(oldValue)
            # add it to my pile of inputs
            self.pyre_bindInputs(newValue)

        # if {trait} is an output
        if trait.output:
            # ask it to forget me
            self.pyre_unbindOutputs(oldValue)
            # tell it i'm one of its factories
            self.pyre_bindOutputs(newValue)
        # chain up
        return super().pyre_traitModified(trait=trait, new=new, old=old)


    # private data
    pyre_inputTraits = ()
    pyre_outputTraits = ()


# end of file
