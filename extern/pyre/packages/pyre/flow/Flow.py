# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# external
import itertools
# superclass
from .Producer import Producer


# declaration
class Flow(Producer, family="pyre.flow"):
    """
    A container of flow nodes
    """


    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Provide a default implementation
        """
        # use the default container
        from .Workflow import Workflow
        # and publish it
        return Workflow


    @classmethod
    def pyre_normalize(cls, descriptor, value, node, **kwds):
        """
        Help convert {value} into a flow instance
        """
        # if {value} is not a string
        if not isinstance(value, str):
            # wouldn't know what to do
            return value

        # otherwise, we will build a flow using {value} as its name; get the default workflow
        # registered with the {descriptor}
        workflow = descriptor.default
        # if it is still my default factory
        if workflow == cls.pyre_default:
            # invoke it
            workflow = workflow()
        # if it is a foundry
        if isinstance(workflow, cls.foundry):
            # invoke it
            workflow = workflow()

        # finally, if it is a component class
        if isinstance(workflow, cls.actor):
            # get the executive
            executive = cls.pyre_executive
            # and the nameserver
            ns = executive.nameserver
            # grab the node meta-data
            info = ns.getInfo(node.key)
            # extract the locator
            locator = info.locator
            # and the priority
            priority = info.priority
            # ask the executive to look for configuration sources based on the flow name
            executive.configure(stem=value, locator=locator, priority=type(priority))
            # instantiate the workflow and return it
            return workflow(name=value, locator=locator)

        # otherwise, leave alone; the validator will check and complain if necessary
        return value


# end of file
