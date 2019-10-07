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
    # instantiate privately
    c = private(name=None)
    # ask for the instance locator
    loc = c.pyre_where()
    # check that it is correct
    assert loc.source == './component_instance_private_locators.py'
    assert loc.line == 26
    assert loc.function == 'test'

    # ask for the trait value and verify it is the default
    assert c.p == "p"
    # ask for its locator
    loc = c.pyre_where('p')
    # check it is correct
    assert loc.source == '<unknown>'

    # set it
    c.p = "q"
    # verify the setting worked
    assert c.p == "q"
    # ask for its locator
    loc = c.pyre_where('p')
    # check it is correct
    assert loc.source == '<unknown>'

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
