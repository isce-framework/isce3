#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify the node layout
"""


def test():
    # access the package
    import pyre.algebraic

    # the algebra
    algebra = pyre.algebraic.algebra
    # declare a node class
    class node(metaclass=algebra): pass


    # verify that the {mro} is what we expect
    assert node.__mro__ == (
        node,
        algebra.base,
        algebra.arithmetic, algebra.ordering, algebra.boolean,
        object)

    # check literals
    assert node.literal.__mro__ == (
        node.literal, algebra.literal, algebra.leaf,
        node,
        algebra.base,
        algebra.arithmetic, algebra.ordering, algebra.boolean,
        object)

    # check variables
    assert node.variable.__mro__ == (
        node.variable, algebra.variable, algebra.leaf,
        node,
        algebra.base,
        algebra.arithmetic, algebra.ordering, algebra.boolean,
        object)

    # check operator
    assert node.operator.__mro__ == (
        node.operator, algebra.operator, algebra.composite,
        node,
        algebra.base,
        algebra.arithmetic, algebra.ordering, algebra.boolean,
        object)

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
