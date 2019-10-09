#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise basic component class configuration
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
    # declare the components
    component = declare()

    # print out the configuration
    # import pyre
    # for error in pyre.executive.errors: print(error)
    # pyre.executive.nameserver.dump(pattern='sample')
    # print("component.p1 = {.p1!r}".format(component))
    # print("component.p2 = {.p2!r}".format(component))

    # check that the settings were read properly
    assert component.p1 == "sample - p1"
    assert component.p2 == "sample - p2"
    # and return the component
    return component


# main
if __name__ == "__main__":
    test()


# end of file
