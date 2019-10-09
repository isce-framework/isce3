#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the various ways traits can pick up their default values
"""

import pyre


def declare(baseFamily=None, baseDefault=0, derivedFamily=None, derivedDefault=""):
    """
    Declare a pair of components
    """
    # the declaration
    class base(pyre.component, family=baseFamily):
        """a component"""
        b = pyre.properties.int(default=baseDefault)

    class derived(base, family=derivedFamily):
        """a derived component"""
        d = pyre.properties.str(default=derivedDefault)

    # return the pair to the caller
    return base, derived


def test():

    # build a pair without family names
    base, derived = declare()
    # check
    assert base.b == 0
    assert derived.b == 0
    assert derived.d == ""

    # now modify the base class property
    base.b = 1
    # check again
    assert base.b == 1
    assert derived.b == 1
    assert derived.d == "" # no cross-talk

    # build a pair with family names that match the sample configuration file
    base, derived = declare(
        baseFamily="sample.inventory.base",
        derivedFamily="sample.inventory.derived")
    # check
    assert base.b == 1
    assert derived.b == 2
    assert derived.d == "Hello world!"

    # modify the base class property
    base.b = 0
    # check
    assert base.b == 0
    assert derived.b == 2
    assert derived.d == "Hello world!"

    # adjust the string property
    derived.d = "{sample.file}"
    # and check
    assert derived.d == "sample.pml"

    return declare


# main
if __name__ == "__main__":
    test()


# end of file
