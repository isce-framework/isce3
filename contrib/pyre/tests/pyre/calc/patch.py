#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify a somewhat non-trivial evaluation network with a mix of node operations
"""


def test():
    import pyre.calc

    # the nodes
    production = pyre.calc.var(value = 80.)
    shipping = pyre.calc.var(value = 20.)
    cost = production + shipping
    margin = .25*cost
    overhead = .45*cost
    price = cost + margin + overhead
    discount = .2
    total = price*(1.0 - discount)
    # check we got the answer right
    assert total.value == 136
    # the poser
    poser = pyre.calc.var(value=180.)

    # introduce the cast
    # print("production: node@{:#x}".format(id(production)))
    # print("  shipping: node@{:#x}".format(id(shipping)))
    # print("      cost: node@{:#x}".format(id(cost)))
    # print("    margin: node@{:#x}".format(id(margin)))
    # print("  overhead: node@{:#x}".format(id(overhead)))
    # print("     price: node@{:#x}".format(id(price)))
    # print("  discount: node@{:#x}".format(id(discount)))
    # print("     total: node@{:#x}".format(id(total)))
    # print("     poser: node@{:#x}".format(id(poser)))

    # patch cost with the new production node
    poser.replace(production)
    # check
    assert cost.value == poser.value + shipping.value
    assert margin.value == .25*cost.value
    assert overhead.value == .45*cost.value
    assert price.value  == cost.value + margin.value + overhead.value
    assert total.value == price.value*(1.0 - discount)
    # check we got the new answer right
    assert total.value == 272

    # make an adjustment
    poser.value = 80
    # check
    assert cost.value == poser.value + shipping.value
    assert margin.value == .25*cost.value
    assert overhead.value == .45*cost.value
    assert price.value  == cost.value + margin.value + overhead.value
    assert total.value == price.value*(1.0 - discount)
    # check we got the answer right
    assert total.value == 136

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
