# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .Node import Node


class Package(Node):
    """
    Handler for the package tag in pml documents
    """

    # constants
    elements = ("component", "package", "bind")


    # interface
    def notify(self, parent, locator):
        """
        Transfer all the key,value bindings to my parent
        """
        # dispatch the assignment events to my parent
        for event in self.assignments:
            parent.assignment(event)

        # dispatch the assignment events to my parent
        for event in self.conditionals:
            parent.conditionalAssignment(event)

        # all done
        return


    # assignment handler
    def assignment(self, event):
        """
        Process a binding of a property to a value
        """
        # add my namespace to the event key
        event.key = self.name + event.key
        # store it with my other bindings
        self.assignments.append(event)
        # and return
        return


    def conditionalAssignment(self, event):
        """
        Process a conditional assignment
        """
        # update the event with my name space
        event.component = self.name + event.component
        event.conditions = [ (self.name+name, family) for name, family in event.conditions ]

        # store it with my other conditional bindings
        self.conditionals.append(event)
        # and return
        return


    # meta methods
    def __init__(self, parent, attributes, locator, **kwds):
        super().__init__(**kwds)

        self.name = attributes['name'].split(self.separator)
        # storage for my assignments
        self.assignments = []
        self.conditionals = []
        return


# end of file
