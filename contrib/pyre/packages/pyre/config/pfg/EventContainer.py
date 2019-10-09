# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Event import Event


# the base class for all containers of configuration events
class EventContainer(Event):
    """
    The abstract base class of all containers of configuration events
    """


    # public data
    assignments = None # the container of unconditional assignments
    conditionalAssignments = None # conditional assignments


    # handlers
    def assignment(self, event):
        """
        Process an unconditional assignment
        """
        # add this to my assignments
        self.assignments.append(event)
        # all done
        return


    def conditionalAssignment(self, event):
        """
        Process a conditional assignment
        """
        # add this to my conditional assignments
        self.conditionalAssignments.append(event)
        # all done
        return


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my containers for conditional and unconditional events
        self.assignments = []
        self.conditionalAssignments = []
        # all done
        return


# end of file
