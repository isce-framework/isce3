# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import itertools
# support
import pyre
# protocols
from .Flow import Flow
from .Producer import Producer
from .Specification import Specification
# my superclass
from .Factory import Factory


# class declaration
class Workflow(Factory, family='pyre.flow.workflow', implements=Flow):
    """
    A container of flow products and factories
    """


    # flow hooks
    @pyre.export
    def pyre_make(self, **kwds):
        """
        Invoke this workflow
        """
        # go through my factories
        for factory in self.pyre_factories():
            # and ask each one to do its thing
            factory.pyre_make(**kwds)
        # if all went well, we are done
        return


    # interface
    def pyre_inputs(self):
        """
        Generate the sequence of my input products
        """
        # collect all non-trivial outputs from my factories
        outputs = { product
                    for factory in self.pyre_factories()
                    for product, _ in factory.pyre_outputs()
                    if product is not None }

        # go through my factories
        for factory in self.pyre_factories():
            # and their inputs
            for product, meta in factory.pyre_inputs():
                # if the product is someone's output
                if product in outputs:
                    # skip it
                    continue
                # otherwise, pass it along
                yield product, meta

        # all done
        return


    def pyre_outputs(self):
        """
        Generate the sequence of my output products
        """
        # collect all non-trivial inputs from my factories
        inputs = { product
                   for factory in self.pyre_factories()
                   for product, _ in factory.pyre_inputs()
                   if product is not None }

        # go through my factories
        for factory in self.pyre_factories():
            # go through their outputs
            for product, meta in factory.pyre_outputs():
                # if the product is someone's input
                if product in inputs:
                    # skip it
                    continue
                # otherwise, pass it along
                yield product, meta

        # all done
        return []


    def pyre_factories(self):
        """
        Generate a sequence of my factories
        """
        # get my inventory
        inventory = self.pyre_inventory
        # go through my facilities
        for trait in self.pyre_facilities():
            # is this is a factory
            if issubclass(trait.protocol, Producer):
                # got one; find the associated node
                factory = inventory[trait].value
                # and pass it along
                yield factory
        # all done
        return


# end of file
