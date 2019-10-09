#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify on-the-fly building of nodes using the overloaded operators
"""


def test():
    import pyre.calc

    # free variables
    c = 100.
    s = 20.

    # make some nodes
    cost = pyre.calc.var(value=c)
    shipping = pyre.calc.var(value=s)
    margin = cost / 2
    price = cost + margin + shipping
    profit = price - margin

    # gather them up
    nodes = [ cost, shipping, margin, price, profit ]

    # verify their values
    # print(cost.value, shipping.value, margin.value, price.value, profit.value)
    assert cost.value == c
    assert shipping.value == s
    assert margin.value == .5*c
    assert price.value == c + s + .5*c
    assert profit.value == c + s

    # make some changes
    c = 200.
    s = 40.
    cost.value = c
    shipping.value = s

    # try again
    # print(cost.value, shipping.value, margin.value, price.value, profit.value)
    assert cost.value == c
    assert shipping.value == s
    assert margin.value == .5*c
    assert price.value == c + s + .5*c
    assert profit.value == c + s

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
