#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that component registration interacts correctly with the pyre configurator model
"""

# access
# print(" -- importing pyre")
import pyre
# print(" -- done")


def declare():

    # declare a protocol
    class protocol(pyre.protocol):
        """a protocol"""
        # properties
        p1 = pyre.properties.str()
        p2 = pyre.properties.str()
        # behavior
        @pyre.provides
        def do(self):
            """behave"""

    # declare a component
    class component(pyre.component, implements=protocol):
        """a component"""
        # traits
        p1 = pyre.properties.str(default="p1")
        p2 = pyre.properties.str(default="p2")

        @pyre.export
        def do(self):
            """behave"""
            return "component"

    return component


def test():
    # declare the component and its protocol
    # print(" -- declaring components")
    component = declare()
    # print(" -- done")

    assert component.p1 == 'p1'
    assert component.p2 == 'p2'
    # grab the component parts
    inventory = component.pyre_inventory
    # get the slots
    p1slot = inventory[component.pyre_trait(alias='p1')]
    p2slot = inventory[component.pyre_trait(alias='p2')]
    # compare the values
    assert component.p1 == p1slot.value
    assert component.p2 == p2slot.value

    return



# main
if __name__ == "__main__":
    test()


# end of file
