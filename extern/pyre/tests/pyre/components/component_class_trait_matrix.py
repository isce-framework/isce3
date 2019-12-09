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


def declare(family=None, default=None):
    """
    Declare a component
    """
    # the declaration
    class component(pyre.component, family=family):
        """a component"""
        value = pyre.properties.int(default=default)

    # return it to the caller
    return component


def test():
    # build one with no family name and a default value for the trait
    c1 = declare()
    # check that it got what we expect
    assert c1.value == None

    # build one with no family name and a string
    c2 = declare(default="1")
    # check that it got what we expect
    assert c2.value == 1

    # build one with no family name and a proper instance
    c3 = declare(default=5)
    # check that it got what we expect
    assert c3.value == 5

    # build one with no family name and a reference to another node
    # get the framework nameserver
    ns = pyre.executive.nameserver
    # build a node
    ns["value"] = "5"
    # now build the component
    c4 = declare(default="{value}")
    # check that it got what we expect
    assert c4.value == 5

    # and return
    return c1, c2, c3, c4


# main
if __name__ == "__main__":
    test()


# end of file
