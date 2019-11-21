# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .Node import Node


class Configuration(Node):
    """
    Handler for the top level tag in pml documents
    """

    # constants
    elements = ("component", "package", "bind")


    # interface
    def notify(self, parent, locator):
        """
        Let {parent} now that processing this configuration tag is complete
        """
        return parent.onConfiguration(self)


    # assignment handler
    def assignment(self, event):
        """
        Process the binding of a property to a value
        """
        # add the event to the pile
        self.configuration.append(event)
        # nothing else, for now
        return


    def conditionalAssignment(self, event):
        """
        Process a binding of a property to a value
        """
        # add the event to the pile
        self.configuration.append(event)
        # nothing else, for now
        return


    # meta methods
    def __init__(self, parent, attributes, locator, **kwds):
        super().__init__(**kwds)
        self.configuration = []
        return


# end of file
