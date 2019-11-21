#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise expression formula updates
"""

# externals
import os


def test():
    # access the package
    import pyre.calc
    # set up the model
    model = pyre.calc.model()

    # the nodes
    production = model.variable(value=80.)
    shipping = .25*production
    cost = model.expression("{production}+{shipping}")
    price = model.expression("2*{cost}")
    # register the nodes
    model["production"] = production
    model["shipping"] = shipping
    model["cost"] = cost
    model["price"] = price

    # check they are all dirty
    assert cost.dirty == True
    assert price.dirty == True
    # check the values
    assert cost.value == production.value + shipping.value
    assert price.value == 2 * cost.value
    # they must all be clean now
    assert cost.dirty == False
    assert price.dirty == False

    # make formula changes
    cost.value = "{production} + 2*{shipping}"
    price.value = "3*{cost}"
    # check they are all dirty
    assert cost.dirty == True
    assert price.dirty == True
    # check the values
    assert cost.value == production.value + 2*shipping.value
    assert price.value == 3 * cost.value
    # they must all be clean now
    assert cost.dirty == False
    assert price.dirty == False

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
