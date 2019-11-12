#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise component registration
"""

# access the framework
import pyre


def test():
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
    class component(pyre.component, family="test", implements=protocol):
        """a component"""
        # traits
        p1 = pyre.properties.str(default="p1")
        p2 = pyre.properties.str(default="p2")

        @pyre.export
        def do(self):
            """behave"""
            return "component"

    # fetch the registrar
    registrar = pyre.executive.registrar

    # check that the protocol is correctly registered
    assert protocol in registrar.protocols
    # check that the component is correctly registered
    assert component in registrar.components
    # check that the set of {protocol} implementers is correct
    assert registrar.implementers[protocol] == {component}

    return component


# main
if __name__ == "__main__":
    # run the test
    component = test()
    # verify that all instances of {component} have been destroyed
    assert tuple(pyre.executive.registrar.components[component]) == ()


# end of file
