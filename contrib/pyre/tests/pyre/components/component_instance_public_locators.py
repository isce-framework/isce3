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

# declare a public component
class public(pyre.component, family="sample.publiclocator"):
    """a public component"""
    # traits
    p = pyre.properties.str(default="p")


def test():
    # instantiate publicly
    c = public(name="c")
    # ask for the instance locator
    loc = c.pyre_where()
    # check that it is correct
    assert loc.source == './component_instance_public_locators.py'
    assert loc.line == 26
    assert loc.function == 'test'

    # ask for the trait value and verify it is the default
    assert c.p == "p"
    # ask for its locator
    loc = c.pyre_where('p')
    # check it is correct
    assert loc.source == './component_instance_public_locators.py'
    assert loc.line == 26
    assert loc.function == 'test'

    # set it
    c.p = "q"
    # verify the setting worked
    assert c.p == "q"
    # ask for its locator
    loc = c.pyre_where('p')
    # check it is correct
    assert loc.source == './component_instance_public_locators.py'
    assert loc.line == 44
    assert loc.function == 'test'

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
