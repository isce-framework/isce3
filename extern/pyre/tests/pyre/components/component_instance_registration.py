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
        p1 = pyre.property()
        p2 = pyre.property()
        # behavior
        @pyre.provides
        def do(self):
            """behave"""

    # declare a component
    class component(pyre.component, family="test", implements=protocol):
        """a component"""
        # traits
        p1 = pyre.property()
        p1.default = "p1"

        p2 = pyre.property()
        p2.default = "p2"

        @pyre.export
        def do(self):
            """behave"""
            return "component"

    # fetch the registrar
    registrar = pyre.executive.registrar

    # instantiate the component
    c1 = component()
    # verify that the instance was recorded in the extent
    assert set(registrar.components[component]) == {c1}

    # instantiate another component
    c2 = component()
    # verify that the instance was recorded in the extent
    assert set(registrar.components[component]) == {c1, c2}

    return component


# main
if __name__ == "__main__":
    # run the test
    component = test()
    # give the executive a chance to clean up
    pyre.executive.shutdown()
    # verify that all instances of {component} have been destroyed
    assert tuple(pyre.executive.registrar.components[component]) == ()


# end of file
