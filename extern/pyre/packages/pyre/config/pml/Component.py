# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .Node import Node


class Component(Node):
    """
    Handler for the component tag in pml documents
    """

    # constants
    elements = ("component", "bind")


    # interface
    def notify(self, parent, locator):
        """
        Transfer all the key,value bindings to my parent
        """
        # dispatch my regular assignments
        for event in self.assignments:
            parent.assignment(event)

        # dispatch my conditional assignments
        for event in self.conditionals:
            parent.conditionalAssignment(event)

        return


    def assignment(self, event):
        """
        Process a binding of a property to a value
        """
        # if i have both a name and a family
        if self.name and self.family:
            # the event key has the current full path to the trait

            # for conditional assignments, we need to split this into two parts: the naked name
            # of the actual trait to receive the value, and the namespace within which this
            # assignment is to take place
            key = event.key[-1:]
            namespace = self.name + event.key[:-1]
            # build a conditional assignment
            event = self.ConditionalAssignment(
                component=namespace, condition=(self.name, self.family),
                key=key, value=event.value,
                locator=event.locator)
            # add it to my conditionals
            self.conditionals.append(event)
            # and return
            return

        # otherwise, deduce my qualifier
        qualifier = self.name if self.name else self.family

        # add the qualifier to the event key
        event.key = qualifier + event.key
        # store it with my other bindings
        self.assignments.append(event)
        # and return
        return


    def conditionalAssignment(self, event):
        """
        Process a conditional assignment
        """
        # if i don't have a name
        if not self.name:
            # raise an exception
            raise NotImplementedError("NYI")

        # update the event with my name space
        event.component = self.name + event.component
        event.conditions = [ (self.name+name, family) for name, family in event.conditions ]

        # if I have both a name and a key
        if self.name and self.family:
            # add my constraints to the conditions
            event.conditions.append((self.name, self.family))

        # store it with my other conditional bindings
        self.conditionals.append(event)
        # and return
        return


    # meta methods
    def __init__(self, parent, attributes, locator, **kwds):
        super().__init__(**kwds)
        # storage for my property bindings
        self.assignments = []
        self.conditionals = []

        # extract the attributes
        name = attributes.get('name')
        family = attributes.get('family')
        # split into fields and store
        self.name = name.split(self.separator) if name else []
        self.family = family.split(self.separator) if family else []

        # make sure that at least one of these attributes were given
        if not self.name and not self.family:
            raise self.DTDError(
                description="neither 'name' nor 'family' were specified",
                locator=locator
                )

        return


# end of file
