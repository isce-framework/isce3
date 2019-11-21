#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that the refcount is zero after all nodes have gone out of scope
"""


def test():

    import pyre.calc
    n1 = pyre.calc.var()
    n2 = pyre.calc.var()

    return


# main
if __name__ == "__main__":
    # request debugging support for the pyre.calc package
    pyre_debug = { "pyre.calc" }
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()
    # get access to the Node class
    from pyre.calc.Node import Node
    # verify reference counts
    # print(tuple(Node._pyre_extent))
    assert tuple(Node._pyre_extent) == ()


# end of file
