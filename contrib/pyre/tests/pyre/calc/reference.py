#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that reference nodes correctly reflect the value of their referends
"""


def test():
    import pyre.calc

    # make a node and set its value
    v = 80.
    production = pyre.calc.var(value=v)
    # make a reference
    clone = production.ref()
    # and a reference to the reference
    clone2 = clone.ref()

    # check
    assert production.value == v
    assert clone.value == v
    assert clone2.value == v

    # once more
    v = 100.
    production.value = v
    assert production.value == v
    assert clone.value == v
    assert clone2.value == v

    return


# main
if __name__ == "__main__":
    # request debugging support for the pyre.calc package
    pyre_debug = { "pyre.calc" }
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()
    # verify reference counts
    # for nodes
    from pyre.calc.Node import Node
    # print(tuple(Node._pyre_extent))
    assert tuple(Node._pyre_extent) == ()


# end of file
