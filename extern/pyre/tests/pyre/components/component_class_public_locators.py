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
    # ask for the class locator
    loc = public.pyre_classWhere()
    # check that it is correct
    assert loc.source == './component_class_public_locators.py'
    assert loc.line == 18
    assert loc.function == '<module>'

    # ask for the trait value and verify it is the default
    assert public.p == "p"
    # ask for its locator
    loc = public.pyre_classWhere('p')
    # check it is correct
    assert loc.source == './component_class_public_locators.py'
    assert loc.line == 18
    assert loc.function == '<module>'

    # set it
    public.p = "q"
    # verify the setting worked
    assert public.p == "q"
    # ask for its locator
    loc = public.pyre_classWhere('p')
    # check it is correct
    assert loc.source == './component_class_public_locators.py'
    assert loc.line == 42
    assert loc.function == 'test'

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
