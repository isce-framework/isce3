# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# framework
import pyre
# my superclass
from .FlowMaster import FlowMaster
# the product protocol
from .Specification import Specification


# declaration
class FactoryMaker(FlowMaster):
    """
    The meta-class of flow nodes
    """


    # methods
    def __new__(cls, name, bases, attributes, **kwds):
        """
        Build a new factory record
        """
        # augment the attributes
        attributes["pyre_inputTraits"] = ()
        attributes["pyre_outputTraits"] = ()
        # chain up to build the record
        factory = super().__new__(cls, name, bases, attributes, **kwds)
        # and pass it on
        return factory


    def __init__(self, name, bases, attributes, **kwds):
        """
        Initialize a new factory record
        """
        # chain up
        super().__init__(name, bases, attributes, **kwds)
        # if this is an internal record
        if self.pyre_internal:
            # all done
            return

        # sort my product into inputs and outputs
        inputs, outputs = self.pyre_classifyProducts()
        # and attach them
        self.pyre_inputTraits = tuple(inputs)
        self.pyre_outputTraits = tuple(outputs)
        # all done
        return


    def pyre_classifyProducts(self):
        """
        Sort my products in to inputs and outputs
        """
        # make piles for inputs and outputs
        inputs = []
        outputs = []
        # go through my facilities, looking for products
        for trait in self.pyre_facilities():
            # if this is a product
            if issubclass(trait.protocol, Specification):
                # and it's an input
                if trait.input:
                    # add it to the pile of inputs
                    inputs.append(trait)
                # if it's an output
                if trait.output:
                    # add it to the pile of outputs
                    outputs.append(trait)
        # all done
        return inputs, outputs


# end of file
