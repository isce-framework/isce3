# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .EventContainer import EventContainer


# the top level object that accumulates the configuration events
class Configuration(EventContainer):
    """
    The resting place for all configuration events harvested during the parsing of {pfg}
    configuration files
    """


    # interface
    def events(self):
        """
        Return all harvested conditional and unconditional assignments
        """
        # first the unconditional assignments
        yield from self.assignments
        # then the conditional ones
        yield from self.conditionalAssignments
        # nothing else
        return


# end of file
