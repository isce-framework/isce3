#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify access to component traits and their meta-data
"""

# support
import pyre

# declare a private component
class private(pyre.component):
    """a private component"""
    # traits
    p = pyre.properties.str(default="p")


def test():
    # ask for the class locator
    loc = private.pyre_classWhere()
    # check that it is correct
    assert loc.source == './component_class_private_locators.py'
    assert loc.line == 18
    assert loc.function == '<module>'

    # ask for the trait value and verify it is the default
    assert private.p == "p"
    # ask for its locator
    loc = private.pyre_classWhere('p')
    # check it is correct
    assert loc.source == '<unknown>'

    # set it
    private.p = "q"
    # verify the setting worked
    assert private.p == "q"
    # ask for its locator
    loc = private.pyre_classWhere('p')
    # check it is correct
    assert loc.source == '<unknown>'

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
