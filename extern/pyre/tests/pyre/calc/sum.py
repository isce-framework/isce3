#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify nodes with sum evaluators
"""


def test():
    import pyre.calc

    # set up the values
    p = 80.
    s = 20.
    # make the nodes
    production = pyre.calc.var(value=p)
    shipping = pyre.calc.var(value=s)
    cost = pyre.calc.sum(production, shipping)
    clone = cost.ref()

    # check the dependencies
    assert tuple(cost.operands) == (production, shipping)
    assert tuple(clone.operands) == (cost,)
    # and the dependents
    assert set(production.observers) == {cost}
    assert set(shipping.observers) == {cost}
    assert set(cost.observers) == {clone}
    # check the values
    assert production.value == p
    assert shipping.value == s
    assert cost.value == p + s
    assert clone.value == p + s

    # update the values
    p = 160.
    s = 40.
    production.value = p
    shipping.value = s

    # check again
    assert production.value == p
    assert shipping.value == s
    assert cost.value == p + s
    assert clone.value == p + s

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
