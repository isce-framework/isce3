# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .EventContainer import EventContainer


# the top level object that accumulates the configuration events
class Section(EventContainer):
    """
    The resting place for all scoped configuration events
    """

    # public data
    name = None
    family = None


    # interface
    def assignment(self, event):
        """
        Process an unconditional assignment
        """
        # unpack my state
        name = self.name
        family = self.family
        # if i have both a name and a family
        if name and family:
            # the event has the current full path to the trait

            # for conditional assignments, we must split this into two parts: the naked name of
            # the actual trait which will receive the value, and the namespace within which
            # this assignment takes place
            key = event.key[-1:]
            namespace = name + event.key[:-1]
            # make a conditional assignment
            assignment = self.events.ConditionalAssignment(
                key = key,
                value = event.value,
                component = namespace,
                condition = (name, family),
                locator = event.locator
                )
            # add it to my conditionals
            self.conditionalAssignments.append(assignment)
            # and return
            return

        # otherwise, this assignment is qualified by my name
        event.key = name + event.key
        # save it with my other assignments
        self.assignments.append(event)
        # all done
        return


    def conditionalAssignment(self, event):
        """
        Process a conditional assignment
        """
        # unpack my state
        name = self.name
        family = self.family

        # adjust the event namespace
        event.component = self.name + event.component
        event.conditions = [ (name+eName, eFamily) for eName, eFamily in event.conditions ]

        # if I have both a name and a family
        if name and family:
            # i have constraints of my own to add to the pile
            event.conditions.append((name, family))

        # store this with my other conditionals
        self.conditionalAssignments.append(event)
        # all done
        return


    def notify(self, parent):
        """
        Place my assignments in my parent's context
        """
        # first, my regular assignments
        for event in self.assignments: parent.assignment(event)
        # and then my conditionals
        for event in self.conditionalAssignments: parent.conditionalAssignment(event)
        # all done
        return


    # meta methods
    def __init__(self, token, **kwds):
        # chain up
        super().__init__(**kwds)
        # get my tag and split it on the fragment marker
        spec = (tag.strip() for tag in token.lexeme.split('#'))
        # and extract the scope levels from each one
        spec = tuple(tag.split('.') for tag in spec)
        # if there is only one part to this specification
        if len(spec) == 1:
            # it's my name
            self.name = spec[0]
            # and i have no family
            self.family = []
        else:
            # unpack a pair; anything else is now an error
            self.family, self.name = spec

        # all done
        return



# end of file
