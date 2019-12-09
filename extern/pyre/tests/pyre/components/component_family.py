#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that component family names are recorded correctly
"""


def test():
    import pyre

    class component(pyre.component, family="test.components.trivial"):
        """a trivial component"""

    # check the family
    assert component.pyre_family() == "test.components.trivial"
    # get the package
    package = component.pyre_package()
    # check the package name
    assert package.name == "test"
    # verify that the protocol is registered
    assert component in package.components

    # all done
    return component


# main
if __name__ == "__main__":
    test()


# end of file
