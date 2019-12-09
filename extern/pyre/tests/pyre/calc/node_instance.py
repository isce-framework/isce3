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
    # get the package
    import pyre.calc
    # make a node class
    class node(metaclass=pyre.calc.calculator): pass

    # make a couple
    n1 = node.variable()
    n2 = node.variable()

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file
