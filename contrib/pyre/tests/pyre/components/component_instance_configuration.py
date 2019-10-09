#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that component instances get configured correctly
"""


def declare():
    import pyre

    class component(pyre.component, family="sample.configuration"):
        """a test component"""
        # properties
        p1 = pyre.properties.str(default="p1")
        p2 = pyre.properties.str(default="p2")

        # behaviors
        @pyre.export
        def do(self):
            """behave"""

    return component


def test():
    # get the declaration
    component = declare()
    # instantiate
    c = component(name="c")
    # check that the configuration setting were transferred correctly
    assert c.p1 == "p1 - instance"
    assert c.p2 == "p2 - instance"
    # and return the component instance
    return c


# main
if __name__ == "__main__":
    test()


# end of file
